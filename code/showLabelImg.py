"""supply a function to label the image and show them"""
"""you just need to point out the image dir and label dir,then you will get the labeled image"""

import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

pic_high = 512
pic_width = 512

def encode_squre(center_c,center_r,width,high):
    """
    trainsfrom to left top position
    :return:transfromed squre
    """
    center_c = float(center_c)
    center_r = float(center_r)
    width = float(width)
    high = float(high)
    center_r = int(pic_high*center_r)
    center_c = int(pic_width*center_c)
    width = int(width*pic_width)
    high = int(high*pic_high)
    ltc = center_c-width//2
    ltr = center_r-high//2
    code = [ltc,ltr,width,high]
    return (" ".join(map(str,code)))


def make_mask(img='',file:str='', shape: tuple = (512, 512),mode='normal',color=(0, 255, 0)):
    """draw mask of img

    Arguments:
        img {cv.img} -- original image
        file {str} -- label of img

    Keyword Arguments:
        shape {tuple} -- shape of image (default: {(512, 512)})
        color {tuple} -- color of squre
        mode {str} -- mask mode :normal of mask
    Returns:
        return label image
    """

    masks = []
    with open(file,"r") as f:
        for line in f:
            masks.append(encode_squre(*line.split()[1:]))
    masks = ";".join(masks)
    # rvt = np.zeros((shape[0], shape[1]))
    label = masks.split(';')
    # img = np.zeros(shape)
    if mode == 'normal':
        for mask in label:
            x,y,w,h = map(int,mask.split(' '))
            cv2.rectangle(img, (x, y), (x + w, y + h), color,1)
        return img
    elif mode == 'mask':
        rvt = np.zeros((shape[0], shape[1],1))
        for mask in label:
            x,y,w,h = map(int,mask.split(' '))
            cv2.rectangle(rvt[:,:,0], (x, y), (x + w, y + h), (255, 255, 255),-1)
        return rvt/255


def labelImg(test_dir="image_train",label_dir="labels_train",output_dir="test_img_output",show=False,color=(0,255,0)):
    """label given images,the most important function here!!!

    Keyword Arguments:
        test_dir {str} -- input dir (default: {"image_train"})
        label_dir {str} -- label dir (default: {"labels_train"})
        output_dir {str} -- output dir (default: {"test_img_output"})
        show {bool} -- show the label image or not (default: {False})
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = os.listdir(test_dir)
    # for i in range(0,len(files),100):
    for file in files:
        # file = files[i]
        if os.path.exists(f"{label_dir}/{file[:-4]}.txt"):
            im = cv2.imread(f"{test_dir}/{file}")
            im = make_mask(im,f"{label_dir}/{file[:-4]}.txt",color=color)
            if show is True:
                plt.imshow(im)
                plt.show()
            cv2.imwrite(f"{output_dir}/{file}",im)



if __name__ == "__main__":

    labelImg(test_dir='../input/test',label_dir="../output",show=False)