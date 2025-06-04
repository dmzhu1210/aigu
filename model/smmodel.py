import dgl
import torch as th
import torch.nn as nn
from model.dpcnn import DPCNN

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.O_f = nn.Linear(h_size, h_size, bias=False)
        self.Z_f = nn.Linear(h_size, h_size, bias=False)
        self.Q_f = nn.Linear(h_size, h_size, bias=False)
        self.K_f = nn.Linear(h_size, h_size, bias=False)
        self.v_f = nn.Linear(h_size, h_size, bias=False)
        self.V_f = nn.Linear(h_size, 1, bias=False)
        self.W_a = nn.Linear(h_size, h_size, bias=False)
        self.W_d = nn.Linear(h_size, h_size, bias=False)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 'e': edges.data['type']}
    def reduce_func(self, nodes):
        et_iou = nodes.mailbox['e'].unsqueeze(-1)
        x_tild = nodes.data['iou']
        n_tild = nodes.data['x']
        h_tild = nodes.mailbox['h']
        # compute attention score
        n_tild = self.Q_f(n_tild)
        h_tild = self.K_f(h_tild)
        v_tild = self.v_f(h_tild)
        w = th.softmax(th.tanh(self.V_f(n_tild.unsqueeze(1) + h_tild)), dim=1)
        modify_score = th.sum(et_iou, dim=1) / et_iou.size(1)
        score_a = (1 - modify_score).unsqueeze(1)
        score_b = modify_score.unsqueeze(1)
        # h_til1 = th.sum(w * et_iou * h_tild, dim=1)
        h_til1 = th.sum((w * score_a * et_iou) * (self.W_a(v_tild)), dim=1)
        h_til0 = th.sum((w * score_b * (1 - et_iou)) * (self.W_d(v_tild)), dim=1)
        # h_til0 = th.sum(w * (1 - et_iou) * h_tild, dim=1)
        h_tild = th.cat([h_til0, h_til1], dim=1)
        f = th.sigmoid(self.O_f(nodes.mailbox['h'] * et_iou) + self.Z_f(nodes.mailbox['h'] * (1-et_iou)))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild) + x_tild, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 loop_nums=5,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.x_size = x_size
        self.h_size = h_size
        self.loop_nums = loop_nums
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size)
        self.dpcnn = DPCNN(in_channel=loop_nums, channel_size=128, emb_dim=h_size, num_classes=2)
        
    def forward(self, graph, cuda=False):
        outputs = []
        graph, features, etype, ntype, mask = graph, graph.ndata['feat'], graph.edata['type'], graph.ndata['type'], graph.ndata['mask']
        graph = graph.to(th.device('cuda:0'))
        features = features.cuda()
        xo = features
        etype = etype.cuda()
        ntype = ntype.cuda()
        ast = etype == 0
        ddgf = etype == 1
        ddgfs = etype == 3
        ddgb = etype == 2
        
        ho = h = th.zeros((graph.num_nodes(), self.h_size), device=features.device)
        co = c = th.zeros((graph.num_nodes(), self.h_size), device=features.device)
        ntype = ntype.unsqueeze(-1)
        
        graph.ndata['iou'] = self.cell.W_iou(self.dropout(features))
        graph.ndata['h'] = h
        graph.ndata['c'] = c
        graph.ndata['x'] = features
        
        step1 = dgl.edge_subgraph(graph, ast + ddgf + ddgfs)
        step2 = dgl.edge_subgraph(graph, ddgb)
        step3 = dgl.edge_subgraph(graph, ast + ddgf)
        
        step1.edata['type'][step1.edata['type']==3] = 1
        step2.edata['type'][step2.edata['type']==2] = 1
        step3.edata['type'][step3.edata['type']==3] = 1
        
        # get original node ids
        step1_ids = step1.ndata[dgl.NID]
        step2_ids = step2.ndata[dgl.NID]
        step3_ids = step3.ndata[dgl.NID]
        
        for idx in range(self.loop_nums):
            if idx == 0:
                step1.ndata['h'] = graph.ndata['h'][step1_ids]
                dgl.prop_nodes_topo(step1,
                                    message_func=self.cell.message_func,
                                    reduce_func=self.cell.reduce_func,
                                    apply_node_func=self.cell.apply_node_func)
           
                graph.ndata['h'][step1_ids] = step1.ndata['h']
                outputs.append(graph.ndata['h'].view(-1, 500, self.x_size))
                
            else:
                step3.ndata['h'] = graph.ndata['h'][step3_ids]
                dgl.prop_nodes_topo(step3,
                                    message_func=self.cell.message_func,
                                    reduce_func=self.cell.reduce_func,
                                    apply_node_func=self.cell.apply_node_func)
           
                graph.ndata['h'][step3_ids] = step3.ndata['h']
                outputs.append(graph.ndata['h'].view(-1, 500, self.x_size)) 
                
            step2.ndata['h'] = graph.ndata['h'][step2_ids]
            step2.ndata['x'] = graph.ndata['x'][step2_ids]
            dgl.prop_nodes_topo(step2,
                                message_func=self.cell.message_func,
                                reduce_func=self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)

            graph.ndata['h'][step2_ids] = step2.ndata['h']

            features = graph.ndata['h'] * ntype + xo * (1 - ntype)
            graph.ndata['h'] = ho
            graph.ndata['c'] = co
            graph.ndata['iou'] = self.cell.W_iou(features)
            graph.ndata['x'] = features
            
        outputs = th.stack(outputs, dim=1)
        outputs = self.dpcnn(outputs)
        return outputs

