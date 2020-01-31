import torch
import torch.nn as nn
import torch.functional as F
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


def ConvLayer(in_c, out_c, k_size, stride=1, upsample=None, instance_norm=True, relu=True,trainable=False):
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(nn.ReflectionPad2d(k_size // 2))
    layers.append(nn.Conv2d(in_c, out_c, k_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_c))
    if relu:
        layers.append(nn.ReLU())
    if trainable:
        layers.append(nn.Conv2d(in_c,out_c,k_size,stride))
    elif not trainable:
        layers.append(MyConv2D(in_c,out_c,k_size,stride))
    return layers


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, k_size=3, stride=1),
            *ConvLayer(channels, channels, k_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x


class TransformNet(nn.Module):
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, k_size=9,trainable=True),
            *ConvLayer(base, base * 2, k_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, k_size=3, stride=2)
        )

        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, k_size=3, upsample=2),
            *ConvLayer(base * 2, base, k_size=3, upsample=2),
            *ConvLayer(base, 3, k_size=9, instance_norm=False, relu=False,trainable=True)
        )

    def forward(self, x):
        y = self.downsampling(x)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y

class MetaNet(nn.Module):
    def __init__(self,para_dict):
        super(MetaNet, self).__init__()
        self.para_num = len(para_dict)
        self.hidden = nn.Linear(1920,128*self.para_num)
        self.fc_dict = {}
        for i,(name,params) in enumerate(para_dict.items()):
            self.fc_dict[name] = i
            setattr(self,'fc{}'.format(i+1),nn.Linear(128,params))

    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self,'fc{}'.format(i+1))
            filters[name] = fc(hidden[:,i*128:(i+1)*128])
        return filters