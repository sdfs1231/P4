import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

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