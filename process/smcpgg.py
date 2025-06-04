# 基于代码状态建模的漏洞检测方法 Vulnerability Detection Method Based on Code State Modeling
import os
import json
import re
import networkx as nx
from glob import glob
from tqdm import tqdm
from utils.tools import *
import networkx as nx
from func_timeout import func_set_timeout
len_file = open('len_file.txt', 'w')
err_file = open('err_file.txt', 'w')
edge_type = {
    'AST': 0,
    'DDG': 1,
    'CFG': 2,
    'self_loop': 3
}

def ast_en(edges, nodes, file):
    etype = list()
    sns, dns = list(), list()
    for e in edges:
        sn, dn = e[0], e[1]
        etype.append(0)
        sns.append(nodes.index(sn))
        dns.append(nodes.index(dn))
    # g = dgl.graph(([], []), num_nodes=5)
    # g.add_edges(sns, dns)
    # try:
    #     dgl.traversal.topological_nodes_generator(g)
    # except:
    #     rev = nx.DiGraph()
    #     for s, n in zip(sns, dns):
    #         rev.add_edge(s, n)
    #     c = list(nx.simple_cycles(rev))
    #     print('find', file)
    return etype, [sns, dns]

def lab2n(edges, n2l, n2c, file):
    l2n = dict()
    n2n = dict()
    paraml = min(list(n2l.values())) # 收行代码的行号
    labs = set()
    labc = set()
    n2labs = dict()
    nedges = list()
    optn = list()
    lab2ns = dict()
    for e in edges:
        sn, dn, lab = e[0], e[1], e[2]
        n2labs[(dn, lab)] = list()
        lab2ns[lab] = list()
    for e in edges:
        sn, dn, lab = e[0], e[1], e[2]
        # 统计依赖项
        labs.add(lab)
        # 判断与变量修改相关的节点，参数视为满足条件的节点
        if opt_nodes(n2c[sn], lab) or n2l[sn] == paraml:
            # 记录不同目标节点在不同依赖项下的源节点
            n2labs[(dn, lab)].append(sn)
            # 记录在某依赖项下满足修改变量节点条件的节点信息
            optn.append((sn, lab))
            # 统计涉及到修改的依赖项
            labc.add(lab)
            # 统计该依赖项所涉及的节点
            lab2ns[lab].append(sn)
        else:
            # 如果此时该依赖项还未找到最近的修改，则将此时的源节点视为修改节点，近邻原则
            # 出现此情景的原因可能是外部变量或者全局变量的引入，这使得该变量并未在单个函数中出现修改和定义
            if lab not in labc:
                n2labs[(dn, lab)].append(sn)
                labc.add(lab)
                optn.append((sn, lab))
                lab2ns[lab].append(sn)
    ks = list()
    # 某些节点并不存在对应的修改节点或不存在边，则不再做处理。
    for k, v in n2labs.items():
        if len(v) == 0:
            ks.append(k)
    for k in ks:
        del n2labs[k]
    # 遍历所有的边
    for e in edges:
        sn, dn, lab = e[0], e[1], e[2]
        # 同行依赖不再做处理, 因为它和单行ast的更新复杂度都相对简单。处理只会增加处理的复杂性和增加时间消耗。
        # 也不对反向ddg进行修改，这是因为反向依赖关系相对简单，其连续依赖仍然是通过正向依赖所体现的，只要正向依赖被化简就足够简单了。同时还能造成自我loop的问题。
        if n2l[sn] == n2l[dn]:
            continue 
        # 由于提前对ddg进行了排序并忽略了反向依赖，每次遍历都可以保证该前节点之前的所有相关节点可以被更新。
        # 通过查找任意目标节点的源节点的源节点是否存在修改节点，可以直接将目标节点与修改节点产生链接，忽略当前源节点，有效的缩短路径。
        if (sn, lab) in n2labs.keys():
            sn_ = n2labs[(sn, lab)]
            n2labs[(dn, lab)] = sn_  
    n2labs = dict(sorted(n2labs.items(), key=parse_line2))
    for e in edges:
        sn, dn, lab = e[0], e[1], e[2]
        try:
            # 对于标量，认为它们在代码中的表达和在词嵌入中的表达都是固定不变的，不受到其他因素的影响，所以切断这些标量的数据依赖，他们对于数据依赖信息的传递本身没有太大的意义。
            int(lab)
            continue
        except:
            pass
        if n2l[sn] == n2l[dn]:
            nedges.append((sn, dn, lab))
            continue
        # 如果边的源点就是修改节点，该边就是最终节点，不做修改。
        if (sn, lab) in optn:
            nedges.append((sn, dn, lab))
            continue
        # 若被标记需要重定向并且本身不是修改变量才进行重定向
        if (sn, lab) in n2labs.keys() and (sn, lab) not in optn:
            # 开始修改需要重定向的节点
            sn_ = n2labs[(sn, lab)]
            # 由于一个节点可能有着多个修改节点，因此重定向时也需要考虑多个。
            for sn__ in sn_:
                # 如果重定向也还是行内节点，则不再修改。
                if n2l[sn__] == n2l[dn]:
                    nedges.append((sn, dn, lab))
                    continue
                # 否则重定向
                nedges.append((sn__, dn, lab))
        else:# (sn, lab) not in n2labs.keys()
            # 有待商榷
            # 还存在一些特殊情况，即变量没有涉及到任何的修改只是参与简单的传递依赖或因为joern本身的问题，没有与之对应的依赖边。按照近邻原则给他分配。
            # 这个操作还会使得同行内的数据依赖，父节点直接接受来自外部的数据而绕过子节点，进一步缩短路径长度。
            tag = False
            for k, v in n2labs.items():
                if k[1] == lab and int(sn) > int(k[0]):
                    sn_ = n2labs[(k[0], lab)]
                    for sn__ in sn_:
                        if n2l[sn__] == n2l[dn]:
                            nedges.append((sn, dn, lab))
                            continue
                        nedges.append((sn__, dn, lab))
                    tag = True
                    break
            if not tag:
                for sn_ in list(reversed(lab2ns[lab])):
                    if int(sn) > int(sn_):
                        nedges.append((sn, dn, lab))
                        break
    # content = []
    # content2 = []
    # snodes1 = set()
    # snodes2 = set()
    # for e in nedges:
    #     s, n, lab = e[0], e[1], e[2]
    #     content.append(f'"{s}" -> "{n}" [ label = "DDG: {lab}"]' + '\n')
    #     snodes1.add(s)
    # for e in edges:
    #     s, n, lab = e[0], e[1], e[2]
    #     content2.append(f'"{s}" -> "{n}" [ label = "DDG: {lab}"]' + '\n')
    #     snodes2.add(s)
    # with open('func2.dot', 'w') as f:
    #     f.write(''.join(content))
    # with open('func3.dot', 'w') as f:
    #     f.write(''.join(content2))
    return nedges, optn
        
# ddg的化简
def ddg_en(edges, nodes, n2l, n2c, n2te, file):
    etype = list()
    sns, dns = list(), list()
    n2adj = dict()
    for n in nodes:
        n2adj[n] = []
    # 统计节点的出边节点
    for e in edges:
        sn, dn, lab = e[0], e[1], e[2]
        if n2l[sn] == n2l[dn]:
            n2adj[sn].append(dn)      
    nedges, optn = lab2n(edges, n2l, n2c, file)       
    for e in nedges:
        sn, dn, lab = e[0], e[1], e[2]
        if sn == dn:
            etype.append(2)
            sns.append(nodes.index(dn))
            dns.append(nodes.index(sn))
            continue
        if n2l[sn] == n2l[dn]:
            if int(sn) < int(dn):
                if (sn, lab) in optn and n2te[sn] == 0:
                    etype.append(3)
                else:
                    etype.append(1)
            else:
                if dn in n2adj[sn]:
                    etype.append(2)
                else:
                    if (sn, lab) in optn and n2te[sn] == 0:
                        etype.append(3)
                    else:
                        etype.append(1)
        else:
            if int(sn) < int(dn):
                if (sn, lab)in optn and n2te[sn] == 0:
                    etype.append(3)
                else:
                    etype.append(1)
            else:
                etype.append(2)
        sns.append(nodes.index(dn))
        dns.append(nodes.index(sn))
    return etype, [sns, dns]
     

def get_etype(edges, nodes, n2l, file, n2c, n2te, keyword=None):
    if keyword is not None:
        if keyword == 'AST':
            etype, edges = ast_en(edges, nodes, file)
        elif keyword == 'DDG':
            etype, edges = ddg_en(edges, nodes, n2l, n2c, n2te, file)
    return etype, edges


def node_is_c(node, cycles):
    if node in cycles:
        return True
    return False

# @func_set_timeout(40)
def get_ntype(nodes, cycles):
    ntype = list()
    n2te = dict()
    for n in nodes:
        if node_is_c(n, cycles):
            ntype.append(1)
            n2te[n] = 1
        else:
            ntype.append(0)
            n2te[n] = 0
    return ntype, n2te

@ func_set_timeout(35)
def get_c(g):
    cycles = list(nx.simple_cycles(g))
    return cycles

def graph_generation(file):
    n2c = dict()
    n2l = dict()
    n2t = dict()
    sns, dns = list(), list()
    ginput = list()
    codes = []
    content = read_file(file)
    len_file = []
    error_file = []
    for_ = False
    if len(content) <=2:
        len_file.append(file)
        return None, None, None, None
    nodes = list()
    ast, ddg, cfgg, ddgg = list(), list(), nx.DiGraph(), nx.DiGraph()
    oriline = list()
    cfgs, cfgd = list(), list()
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isnode:
            nid, lid = isnode.group(1), isnode.group(2)
            code = re.search(codep, line).group(1)
            # if 'for (' in code:
            #     for_ = True
            token = word2vec(code)
            codes.append(code)
            ginput.append(token)
            nodes.append(nid)
            n2c[nid] = code
            n2l[nid] = lid
            # n2t[nid] = token
            oriline.append(line)
        if isedge:
        #     if not for_:
        #         return None, None, None, None
            stod = re.search(atobp, line)
            sn, dn = stod.group(1), stod.group(2)
            if sn not in nodes or dn not in nodes:
                continue
            if 'AST:' in line:
                ast.append((sn, dn))
            elif 'CFG:' in line:
                cfgg.add_edge(sn, dn)
                cfgs.append(nodes.index(sn))
                cfgd.append(nodes.index(dn))
            elif 'DDG:' in line:
                lab = re.search(labep, line)
                lab = lab.group(1)
                lab = lab.split('DDG:')[-1].strip()
                if lab == '':
                    continue
                lab = code_normal(lab)
                ddg.append((sn, dn, lab))
    if len(nodes) > 500:
        return None, None, None, None
    try:
        c = get_c(cfgg)
    except:
        return None, None, None, None
    cy = set()
    for ns in c:
        for n in ns:
            cy.add(n)
    ddg = sorted(ddg, key=parse_line)
    ntype, n2te = get_ntype(nodes, cy)
    astey, aste = get_etype(ast, nodes, n2l, file, n2c, n2te, keyword='AST')
    ddgey, ddge = get_etype(ddg, nodes, n2l, file, n2c, n2te, keyword='DDG')
    test_g = nx.DiGraph()
    for n, s in zip(aste[0], aste[1]):
        test_g.add_edge(n, s)
    for t, n, s in zip(ddgey, ddge[0], ddge[1]):
        if t == 1 or t ==3:
            test_g.add_edge(n, s)
    rev = list(nx.simple_cycles(test_g))
    if len(rev) > 0:
        error_file.append(file)
        # err_file.write(file + '\n')
        return None, None, None, None
    # ginput = pre_word2vec(codes)
    etype = astey + ddgey
    edges = [aste[0] + ddge[0], aste[1] + ddge[1]]

    return ginput, edges, etype, ntype


if __name__ == '__main__':
    # graph_generation('fix/temp1.dot')
    # for folder in ['dataset/fan3']:
    files = glob(os.path.join('dataset/cpgs2', '*.dot'))
    for file in tqdm(files):
        label = 0 if 'novul' in file else 1
        basename = os.path.basename(file)
        ginput, edge, etype, ntype = graph_generation(file)
        if ginput is None:
            continue
        data_point = {
            'node_features': ginput,
            'graph': edge,
            'etype': etype,
            'ntype': ntype,
            'target': label,
        }
            
        with open(os.path.join('dataset/train_diverse', basename.replace('.dot', '.json')), 'w') as f:
            json.dump(data_point, f)