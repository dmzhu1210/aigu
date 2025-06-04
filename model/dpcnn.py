import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    the modules for detection via classify task
"""

class DPCNN(nn.Module):
    def __init__(self, in_channel, channel_size, emb_dim, num_classes):
        super(DPCNN, self).__init__()
        
        self.conv_region1 = nn.Conv2d(in_channel, channel_size//4, (1, emb_dim), stride=1, padding=(0, 0), bias=True)
        self.conv_region3 = nn.Conv2d(in_channel, channel_size//4, (3, emb_dim), stride=1, padding=(1, 0), bias=True)
        self.conv_region5 = nn.Conv2d(in_channel, channel_size//4, (5, emb_dim), stride=1, padding=(2, 0), bias=True)
        self.conv_pool = nn.MaxPool2d(kernel_size=(3, emb_dim), stride=1, padding=(1, 0))
        self.conv_down = nn.Conv2d(in_channel, channel_size//4, kernel_size=(1, emb_dim), stride=1, bias=True)
        
        self.conv1 = nn.Conv2d(channel_size, channel_size//4, (1, 1), stride=1)
        self.conv3 = nn.Conv2d(channel_size, channel_size//4, (3, 1), stride=1, padding=(1, 0))
        self.conv5 = nn.Conv2d(channel_size, channel_size//4, (5, 1), stride=1, padding=(2, 0))
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.down = nn.Conv2d(channel_size, channel_size//4, kernel_size=1, stride=1)
        
        self.pooling = nn.MaxPool2d((3, 1), stride=2)
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channel_size, num_classes)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        bc = x.size()[0]
        
        branch1 = self.conv_region1(x)
        branch2 = self.conv_region3(x)
        branch3 = self.conv_region5(x)
        branch4 = self.conv_pool(x)
        branch4 = self.conv_down(x)
        
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        
        x = self._padd_and_conv(self.relu, self.conv1, self.conv3, self.conv5, self.pool, self.down, x) 
        x = self._padd_and_conv(self.relu, self.conv1, self.conv3, self.conv5, self.pool, self.down, x) 

        while x.size()[2] > 2: 
            x = self._block(x)

        x = x.view(bc, -1)        # [batch_size, channel_size]
        x = self.drop(x)
        x = self.fc(x)            # [batch_size, num_classes]
        return x

    def _block(self, x):
        x = self.padding_pool(x)
        px = self.pooling(x)

        x = self._padd_and_conv(self.relu, self.conv1, self.conv3, self.conv5, self.pool, self.down, px) 
        x = self._padd_and_conv(self.relu, self.conv1, self.conv3, self.conv5, self.pool, self.down, x) 
        
        x = x + px
        return x

    def _padd_and_conv(self, relu, conv1, conv3, conv5, pool, down, x):
        x = relu(x)
        branch1 = conv1(x)
        branch2 = conv3(x)
        branch3 = conv5(x)
        branch4 = pool(x)
        branch4 = down(branch4)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x