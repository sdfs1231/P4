import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from tools import mean_std,Smooth
from MyNet import VGG,TransformNet,MetaNet
from collections import defaultdict

#init params
style_weight = 50
content_weight = 1
tv_weight = 1e-6
epochs = 22
batch_size = 8
width = 256

#init net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained = True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
transformnet = TransformNet(32).to(device)
transformnet.get_param_dict()
metanet = MetaNet(transformnet.get_param_dict()).to(device)


cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)

#init dataset
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder('/home/ypw/WikiArt/', transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('/home/ypw/COCO/', transform=data_transform)

content_data_loader = torch.utils.data.DataLoader(content_dataset,batch_size = batch_size,
                                                  shuffle=True)
#eval net
optimizer = optim.Adam(transformnet.parameters(),lr=1e-3)
metanet.eval()
transformnet.eval()

rands = torch.rand(4,3,256,256).to(device)
features = vgg16(rands)
weights = metanet(mean_std(features))
transformnet.set_weights(weights)
transformed_imgs = transformnet(torch.rand(4,3,256,256).to(device))

trainable_params = {}
trainable_param_shapes = {}
for model in [vgg16, transformnet, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape

n_batch = len(content_data_loader)
metanet.train()
transformnet.train()

for epoch in range(epochs):
    smoother = defaultdict(Smooth)
    with tqdm(enumerate(content_data_loader),total=n_batch) as pbar:
        for batch,(content_imgs,_) in pbar:
            n_iter = epoch*n_batch + batch

            if batch %20 ==0:
                style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
                style_features = vgg16(style_image)
                style_mean_std = mean_std(style_features)

            x = content_imgs.cpu().numpy()
            if (x.min(-1).min(-1)==x.max(-1).max(-1).any()):
                continue

            optimizer.zero_grad()

            weights = metanet(mean_std(style_features))
            transformnet.set_weights(weights,0)

            content_imgs = content_imgs.to(device)
            transformed_imgs = transformnet(content_imgs)

            content_features = vgg16(content_imgs)
            transformed_features = vgg16(transformed_features)
            transformed_mean_std = mean_std(transformed_features)

            content_loss = content_weight * F.mse_loss(transformed_features[2],content_features[2])

            style_loss = style_weight * F.mse_loss(transformed_mean_std,
                                                   style_mean_std.expand_as(transformed_mean_std))

            y = transformed_imgs
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:,:,:,:-1]-y[:,:,:,1:]))+
                                   torch.sum(torch.abs(y[:,:,:-1,:]-y[:,:,1:,:])))

            loss = content_loss + style_loss +tv_loss

            loss.backward()
            optimizer.step()

            smoother['content_loss'] += content_loss.item()
            smoother['style_loss'] += style_loss.item()
            smoother['tv_loss'] += tv_loss.item()
            smoother['loss'] += loss.item()

            max_value = max([x.max().item() for x in weights.values()])

            s = 'Epoch: {} '.format(epoch + 1)
            s += 'Content: {:.2f} '.format(smoother['content_loss'])
            s += 'Style: {:.1f} '.format(smoother['style_loss'])
            s += 'Loss: {:.2f} '.format(smoother['loss'])
            s += 'Max: {:.2f}'.format(max_value)
