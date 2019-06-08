from __future__ import print_function

from .reader import name_file 

import json
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from skimage.transform import resize
from skimage.io import imsave, imread


#----------------------------------------------
#----------------------------------------------
## Step 3
## Pre-processing the images: resizing, filtering, normalizing



def preprocess_ris(imgs,img_cols,img_rows):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def normalizing_imgs(imgs):
    imgs_n=imgs/255
    return imgs_n

def normalizing_mean_imgs(imgs):
    #--------------------------
    #
    ## normalize the images using the mean values
    ## it computes the mean values directly from the X object
    ## for RGB images the mean value is computed for each channel
    ## N.B the mean values are affected by the number of considered images
    #
    #----------------------
    if np.size(imgs.shape)>3:
        norm_R=np.uint8(imgs[:,:,:,0].mean())
        norm_G=np.uint8(imgs[:,:,:,1].mean())
        norm_B=np.uint8(imgs[:,:,:,2].mean())
        imgs[:,:,:,0]-=norm_R
        imgs[:,:,:,1]-=norm_G
        imgs[:,:,:,2]-=norm_B
    else:
        norm=np.uint8(imgs.mean())
        imgs-=norm
    return imgs               


def normalizing_RGB_imgs(imgs,norm_r,norm_g,norm_b):
    #--------------------------
    #
    ## normalize the images using the mean values
    ## that are computed previously
    #
    #----------------------
    imgs[:,:,:,0]-=norm_r
    imgs[:,:,:,1]-=norm_g
    imgs[:,:,:,2]-=norm_b
    return imgs

def normalizing_1ch_imgs(imgs,norm):
    #--------------------------
    #
    ## normalize the images using the mean values
    ## that are computed previously
    #
    #----------------------
    imgs-=norm
    return imgs

def getting_mean_from_images(image_path):
    
    ## for images having one channel
    ## PAN that has 1 channel
    ## 
    images = name_file(image_path)
    total = len(images)

    mean_imag = 0
    i = 0
    print('-'*30)
    print('evaluating mean images...')
    print('-'*30)
    for image_name in images:
        img = imread(os.path.join(image_path, image_name), as_gray=True)
        r=img.mean()
        mean_imag+=r
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    mean_imag=np.uint8(mean_imag/total)

    
def getting_mean_from_RGBimages(image_path):
    
    ## for RGB images having more than one channel
    images = name_file(image_path)
    total = len(images)
   
    mean_imagR = 0
    mean_imagG = 0
    mean_imagB = 0
    i = 0
    print('-'*30)
    print('evaluating mean images...')
    print('-'*30)
    for image_name in images:
        img = imread(os.path.join(image_path, image_name))
        r=img[:,:,0].mean()
        g=img[:,:,1].mean()
        b=img[:,:,2].mean()
        mean_imagR += r
        mean_imagG += g
        mean_imagB += b

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    mean_imagR =  np.uint8(mean_imagR/total)
    mean_imagG =  np.uint8(mean_imagG/total)
    mean_imagB =  np.uint8(mean_imagB/total)
    return mean_imagR, mean_imagG, mean_imagB


