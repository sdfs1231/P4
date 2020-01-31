import torch
import torch.nn as nn
import torch.nn.functional as F
from MyNet import VGG
from gram_matrix import gram_matrix
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from read_image import read_image,recover_image,imshow
import argparse
import torchvision.transforms as transforms
import torchvision.models as models
from read_image import read_image,imshow,recover_image


#net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(pretrained = True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()


#input img
width = 512
content_img = read_image('content_img.jpg',width).to(device)
style_img = read_image('style_img.jpg',width).to(device)
input_img = content_img.clone()

# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# imshow(style_img, title='Style Image')
#
# plt.subplot(1, 2, 2)
# imshow(content_img, title='Content Image')

style_features = vgg16(style_img)
content_features = vgg16(content_img)

style_grams = [gram_matrix(x) for x in style_features]

# train
optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e6
content_weight = 1
run = [0]

while run[0]<300:
    print(run[0],'th training')
    def f():
        optimizer.zero_grad()

        features = vgg16(input_img)

        content_loss = F.mse_loss(features[2],content_features[2]) * content_weight
        style_loss = 0

        grams = [gram_matrix(x) for x in features]

        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight

        loss = style_loss + content_loss

        if run[0] % 50 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                run[0], style_loss.item(), content_loss.item()))
        run[0] += 1

        loss.backward()
        return loss
    optimizer.step(f)

#visualize the transform
plt.ion()
plt.figure(figsize = (18,6))

plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')

plt.subplot(1, 3, 3)
imshow(input_img, title='Output Image')

plt.savefig('1st_stage.jpg')
plt.ioff()