import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from MyNet import VGG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained = True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
width = 256
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder('/home/ypw/WikiArt/', transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('/home/ypw/COCO/', transform=data_transform)

for batch,(content_imgs,_) in pbar:
    if batch %20 == 0:
        style_img = random.choice(style_dataset)[0].unsqueeze(0).to(device)
        style_features = vgg16(style_img)
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
