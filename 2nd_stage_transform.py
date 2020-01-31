import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import random
from MyNet import VGG
from read_image import read_image,imshow
from PIL import Image
from gram_matrix import gram_matrix
from MyNet2 import TransformNet
from tqdm import tqdm
import torch.optim as optim
from tools import Smooth,save_debug_img
import torch.nn.functional as F
import matplotlib.pyplot as plt

#init device @ net @ dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained = True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()
transformnet = TransformNet(32).to(device)
b_size = 4
verbose_batch = 800
width = 256
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
style_weight = 1e5
tv_weight = 1e-6
content_weight = 1
data_transform = transforms.Compose([
    transforms.Resize(width),
    transforms.CenterCrop(width),
    transforms.ToTensor(),
    tensor_normalizer,
])

dataset = torchvision.datasets.ImageFolder('/home/ypw/COCO/',transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=b_size,shuffle=True)

#load imgs
style_img = Image.open('style_img.jpg')

#compute style features
style_features = vgg16(style_img)
style_gram = [gram_matrix(x) for x in style_features]
style_grams = [x.detach() for x in style_gram]

#train
optimizer = optim.Adam(transformnet.parameters(),lr=1e-3)
transformnet.train()

n_batch = len(data_loader)

for epoch in range(1):
    print('Epoch{}'.format(epoch+1))
    smooth_content_loss = Smooth()
    smooth_style_loss = Smooth()
    smooth_tv_loss = Smooth()
    smooth_loss = Smooth()

    with tqdm(enumerate(data_loader),total = n_batch) as pbar:
        for batch,(content_imgs,_) in pbar:
            optimizer.zero_grad()

            content_imgs = content_imgs.to(device)
            transformed_imgs = transformnet(content_imgs)
            transformed_imgs = transformed_imgs.clamp(-3,3) #why? -3 -3?

            content_features = vgg16(content_imgs)
            transformed_features = vgg16(transformed_imgs)

            content_loss = content_weight * F.mse_loss(transformed_features[1], content_features[1])

            y = transformed_imgs
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            style_loss = 0.
            transformed_grams = [gram_matrix(x) for x in transformed_features]

            for transformed_gram,style_gram in zip(transformed_grams,style_grams):
                style_loss += style_weight * F.mse_loss(transformed_gram,
                                                        style_gram.expand_as(transformed_gram))

            loss = style_loss+content_loss+tv_loss
            loss.backward()

            optimizer.step()

            smooth_content_loss += content_loss.item()
            smooth_style_loss += style_loss.item()
            smooth_tv_loss += tv_loss.item()
            smooth_loss += loss.item()

            s = f'Content: {smooth_content_loss:.2f} '
            s += f'Style: {smooth_style_loss:.2f} '
            s += f'TV: {smooth_tv_loss:.4f} '
            s += f'Loss: {smooth_loss:.2f}'
            if batch % verbose_batch == 0:
                s = '\n' + s
                save_debug_img(style_img, content_imgs, transformed_imgs,
                                 f"debug/s2_{epoch}_{batch}.jpg")

            pbar.set_description(s)
        torch.save(transformnet.state_dict(), 'transform_net.pth')

        content_img = random.choice(dataset)[0].unsqueeze(0).to(device)
        output_img = transformnet(content_img)

        plt.ion()
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        imshow(style_img, title='Style Image')

        plt.subplot(1, 3, 2)
        imshow(content_img, title='Content Image')

        plt.subplot(1, 3, 3)
        imshow(output_img.detach(), title='Output Image')

        plt.savefig('2nd_stage.jpg')
        plt.ioff()