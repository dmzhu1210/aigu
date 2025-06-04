from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from model.smmodel import TreeLSTM
import torch.nn as nn
import torch

   

codebert = AutoModelForSequenceClassification.from_pretrained('pretrained/codebert', num_labels=2)    
codebert.load_state_dict(torch.load('pretrain_reveal.bin'))

treelstm = TreeLSTM(x_size=100, h_size=100, num_classes=2, dropout=0.5)
treelstm.load_state_dict(torch.load('my_reveal.bin'))

class joint_model(nn.Module):
    def __init__(self):
        super(joint_model, self).__init__()
        self.codebert = codebert
        self.treelstm = treelstm
        self.weight = nn.Parameter(torch.tensor(0.5))
        # for param in self.codebert.parameters():
        #     param.requires_grad = False
        # for param in self.treelstm.parameters():
        #     param.requires_grad = False
    def forward(self, x, y, m):
        # with torch.no_grad():
        outputa = self.codebert(y, m)
        outputa = outputa.logits
        outputb = self.treelstm(x)
        final = outputb * self.weight + outputa * (1 - self.weight)
        return final
    