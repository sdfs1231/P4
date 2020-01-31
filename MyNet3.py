import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyConv2D(nn.Module):
    def __init__(self,in_c,out_c,k_size=3,stride=1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros((out_c,in_c,k_size,k_size))
        self.bias = torch.zeros(out_c).to(device)

        self.in_c = in_c
        self.out_c = out_c
        self.k_size = k_size
        self.stride = stride

    def forward(self,x):
        return F.conv2d(x,self.weight,self.bias,self.stride)

def ConvLayer(in_c,out_c,k_size,upsample=None,instance_norm=True,)