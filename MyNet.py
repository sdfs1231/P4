import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np

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

    def get_param_dict(self):
        param_dict = defaultdict(int)
        def dfs(module,name):
            for name2 , layer in module.named_children():
                dfs(layer,'%s.%s' % (name,name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))
        dfs(self,'')
        return param_dict

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

class VGG(nn.Module):
    def __init__(self,features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping  = {
            '3':'relu1_2',
            '8':'relu2_2',
            '15':'relu3_3',
            '22':'relu4_3'
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):
        outs = []
        for name,module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg16 = models.vgg16(pretrained = True)
    vgg16 = VGG(vgg16.features[:23]).to(device).eval()
    features = vgg16(input_img) #origin image
    content_features = vgg16(content_img) #style image
    content_loss = F.mse_loss(features[2], content_features[2]) * content_weight