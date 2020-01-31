from PIL import Image
import numpy as np
from read_image import recover_image

width = 256
def tensor_2_array(tensor):
    x = tensor.detach().cpu().numpy()
    x = (x*255).clip(0,255).transpose(0,2,3,1).astype(np.uint8)
    return x

def save_debug_img(style_images, content_images, transformed_images, filename):
    style_img = Image.fromarray(recover_image(style_images))
    content_imgs = [recover_image(x) for x in content_images]
    transformed_imgs = [recover_image(x) for x in transformed_images]
    new_img = Image.new('RGB',(style_img.size[0] + (width+5)*4 , max(style_img.size[1],width*2+5)))
    new_img.paste(style_img,(0,0))

    x = style_img.size[0] + 5
    for i,(a,b) in enumerate(zip(content_imgs,transformed_imgs)):
        new_img.paste(Image.fromarray(a),(x+(width+5)*i,0))
        new_img.paste(Image.fromarray(b),(x+(width+5)*i,width+5))
    new_img.save(filename)

class Smooth():
    def __init__(self,window_size=100):
        self.window_size = window_size
        self.data = np.zeros((window_size,1),dtype=np.float32)
        self.indx = 0

    def __iadd__(self, x):
        if self.indx == 0:
            self.data[:] = x
        self.data[self.indx%self.window_size] = x
        self.indx += 1
        return self

    def __float__(self):
        return float(self.data.mean())

    def __format__(self, f):
        self.__float__().__format__(f)