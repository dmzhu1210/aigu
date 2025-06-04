import argparse
import logging
import os
import sys
import torch.nn as nn
from dataset import train_dataset, test_dataset, valid_dataset, collate_fn, collate_fn2
os.chdir(sys.path[0])
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from trainer import train
from transformers import RobertaModel, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import math
from torch.optim.optimizer import Optimizer
from model.smmodel import TreeLSTM
import torch.optim as optim
# from joint_model import joint_model

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-6,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        beta2_t = None
        ratio = None
        N_sma_max = None
        N_sma = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                if beta2_t is None:
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    beta1_t = 1 - beta1 ** state['step']
                    if N_sma >= 5:
                        ratio = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / beta1_t

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    step_size = group['lr'] * ratio
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / beta1_t
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
    
# class mymodel(nn.Module):
#     def __init__(self):
#         super(mymodel, self).__init__()
#         # self.code = RobertaForSequenceClassification.from_pretrained("pretrained/codebert")
        
#         # self.code.pooler = nn.Sequential(nn.Linear(in_features=768, out_features=200, bias=True), nn.Tanh())
#         # self.linear = nn.Linear(200, 2)
#     def forward(self, x, y):
#         pre = self.code(x, y)
#         # pre = pre.pooler_output[:, 0, :]
#         # pre = self.linear(pre)
#         return pre.logits



if __name__ == '__main__':
    torch.manual_seed(22)
    np.random.seed(22)
    # model = AutoModelForSequenceClassification.from_pretrained('pretrained/codebert', num_labels=2)    
    # model = DevignModel(input_dim=100, output_dim=100,
    #                     num_steps=6, max_edge_types=9)
    # model = mymodel()
    # for param in model.code.embeddings.parameters():
    #     param.requires_grad = False
    # for layer in model.code.encoder.layer[:11]:
    #     for param in layer.parameters():
    #         param.requires_grad = False
            
    model = TreeLSTM(x_size=100, h_size=100, num_classes=2, dropout=0.5, loop_nums=4)
    # model = joint_model()
    # for param in model.treelstm.parameters():
    #     param.requires_grad = False
    # for param in model.codebert.parameters():
    #     param.requires_grad = False

    model.cuda()
    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1.2])).float(),reduction='sum')
    # loss_function = CrossEntropyLoss(reduction='mean')
    loss_function.cuda()
    LR = 2e-4
    
#     train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn2)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn2)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn2)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optim = RAdam(trainable_params, lr=LR, weight_decay=1e-6)
    optim = RAdam(model.parameters(), lr=LR, weight_decay=1e-6)
    print('lr: {}, loop_nums: 5'.format(LR))
    train(model=model, dataset=[train_loader, test_loader, valid_loader], epoches=100, dev_every=len(train_loader),
          loss_function=loss_function, optimizer=optim,
          save_path= './pretrained', max_patience=100, log_every=5)