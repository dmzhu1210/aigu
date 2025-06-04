import torch
from model.smmodel import TreeLSTM

model = TreeLSTM(x_size=100, h_size=100, num_classes=2, dropout=0.5, loop_nums=5)
model.load_state_dict(torch.load('my_reveal.bin'))