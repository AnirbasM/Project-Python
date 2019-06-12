from __future__ import print_function

import json
import rasterio as rio
import numpy as np
import pandas as pd
import os

from shapely.geometry import Point, Polygon
import shapely.wkt

import geopandas as gpd
import sys
import geojson


from skimage.transform import resize
import skimage.draw
from skimage.io import imsave, imread

from sklearn.model_selection import train_test_split



def path_from_AOI_id(AOI_id):

    if AOI_id=='AOI_5':
        Path_AOI_id='AOI_5_Khartoum_Train'
    if AOI_id=='AOI_4':
        Path_AOI_id='AOI_4_Shanghai_Train'
    if AOI_id=='AOI_3':
        Path_AOI_id='AOI_3_Paris_Train'
    if AOI_id=='AOI_2':
        Path_AOI_id='AOI_2_Vegas_Train'
   
    return Path_AOI_id

def name_file(image_path):
    #Select the tiff file inside a directory'
    images=[]
    for file in os.listdir(image_path):
        if file.endswith(".tif"):
            images.append(file)
            
    return images

def name_file_geojson(image_path):
    #Select the geojson file inside a directory'
    images=[]
    for file in os.listdir(image_path):
        if file.endswith(".geojson"):
            images.append(file)
            
    return images

def get_ch_rows_columns(image_path):
    # get the number of channela, rows and columns
    # of the images that are inside the im directory (image_path)
    # it doesn't work with the mask()
    # it takes the first image as example
    images=name_file(image_path)
    if len(images)>0:
        name=os.path.join(image_path, images[1])
    
        with rio.open(name, "r") as src2:
            array = src2.read()
        return array.shape
    else:
        print('no image in the directory')

def get_image_id(image_name):
    #for example, image_name=PAN_AOI_5_Khartoum_img9 takes just AOI_5_Khartoum_img9
    # it is used to get the masks inside the summary table
    x=image_name.split('AOI')
    image_id='AOI'+ x[-1]
    image_id=image_id.split('.tif')[0]
    return image_id 


#open the summary of the polygons observed in each city
def summary_poly_AOI_id(AOI_id):
    image_path=get_image_path(AOI_id,'Sum')
    os.listdir(image_path) 
    for file in os.listdir(image_path):
        if file.endswith(".csv"):
            name_file= image_path+file
            df=pd.read_csv(name_file)     
    return df

## image path


#MUL_path    = path_from_AOI_id(AOI_id)+'/MUL/'
#PAN_path    = path_from_AOI_id(AOI_id)+'/PAN/'
#MUL_ps_path = path_from_AOI_id(AOI_id)+'/MUL-PanSharpen/'
#RGB_path    = path_from_AOI_id(AOI_id)+'/RGB-PanSharpen/'
#mask_path   = path_from_AOI_id(AOI_id)+'/geojson/buildings/'


def get_image_path(AOI_id,fmt):
    if fmt=='MUL':
        image_path     = path_from_AOI_id(AOI_id)+'/MUL/'
    if fmt=='PAN':
        image_path     = path_from_AOI_id(AOI_id)+'/PAN/'
    if fmt=='PAN_sh':
        image_path    = path_from_AOI_id(AOI_id)+'/MUL-PanSharpen/'
    if fmt=='RGB':
        image_path     = path_from_AOI_id(AOI_id)+'/RGB-PanSharpen/'
    if fmt=='mask':
        image_path     = path_from_AOI_id(AOI_id)+'/geojson/buildings/'
    if fmt=='Sum':
        image_path     = path_from_AOI_id(AOI_id)+'/summaryData/'
    return image_path

def loading_data():
    X= np.load('data/X.npy')
    Y= np.load('data/Y.npy')
    Y_order= np.load('data/Y_order.npy')  
    return X, Y, Y_order

def splitting_data_test_train():
    X,y, Y_order = loading_data()
    X_train, X_test, y_train, y_test, indices_train,indices_test = train_test_split(
    X, y, Y_order, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, indices_train,indices_test 


def loading_test_data():
    Y_test = np.load('data/y_test.npy')
    indices_Y= np.load('data/indices_test.npy')
    return Y_test, indices_Y


def loading_train_data():
    X_train = np.load('data/X_train.npy')
    Y_train = np.load('data/y_train.npy')
    return X_train, Y_train

## --------Preparing data--------------#
## Step 2.1
## Saving the images in a np.array --> X_train (n_images, nrows,ncolumn) 
## ------------------------------------


def create_x_train_data(image_path):
    
    ## for images having one channel
    ## PAN that has 1 channel
    ## 
    images = name_file(image_path)
    total = len(images)
    ch,rows,cols=get_ch_rows_columns(image_path)

    imgs = np.ndarray((total, rows, cols), dtype=np.uint8)
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # 1. reading the image with imread
        # 2. transforming in a np.array with np.array([])
        # 3. saving the array in the i position of the imgs(Nimages x rows x columns)
        img = imread(os.path.join(image_path, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    #np.save('imgs_train.npy', imgs)
    #np.save('imgs_mask_train.npy', imgs_mask)
    #print('Saving to .npy files done.')
    return imgs

def create_x_train_data_RGB(image_path):
    
    ## for images having more than one channel
    ## RGB that has 3 channel
    ## 
    images = name_file(image_path)
    total = len(images)
    ch,rows,cols=get_ch_rows_columns(image_path)

    imgs_RGB = np.ndarray((total, rows, cols,ch), dtype=np.uint8)
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # 1. reading the image with imread
        # 2. transforming in a np.array with np.array([])
        # 3. saving the array in the i position of the imgs(Nimages x rows x columns)
        img = imread(os.path.join(image_path, image_name))
        
        img = np.array([img])
        imgs_RGB[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    #np.save('imgs_train.npy', imgs)
    #np.save('imgs_mask_train.npy', imgs_mask)
    #print('Saving to .npy files done.')
    return imgs_RGB

def create_x_train_data_MUL8(image_path):
    
    ## for images having more than one channel
    ## Mul with 8 channels 
    ## 
    images = name_file(image_path)
    total = len(images)
    ch,rows,cols=get_ch_rows_columns(image_path)

    imgs_MUL8 = np.ndarray((total, rows, cols,ch), dtype=np.uint8)
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # 1. reading the image with imread
        # 2. transforming in a np.array with np.array([])
        # 3. saving the array in the i position of the imgs(Nimages x rows x columns)
        img = imread(os.path.join(image_path, image_name))
        
        img = np.array([img])
        imgs_RGB[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    #np.save('imgs_train.npy', imgs)
    #np.save('imgs_mask_train.npy', imgs_mask)
    #print('Saving to .npy files done.')
    return imgs_MUL8



## --------Preparing data--------------#
## Step 2.2
## Saving the images in a np.array --> X_train (n_images, nrows,ncolumn) 
## ------------------------------------

def create_mask_im(df,image_id):
    #for each image_id create a mask based on the polygons
    # described in the summary table
    im_mask = np.zeros((650, 650))
    
    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    im_mask = (im_mask > 0.5).astype(np.uint8)
    return im_mask


def create_y_train_data(image_path,df):
    # takes the image_id of the tiff images and 
    # check if they can be found in df
    
    images = name_file(image_path)
    total = len(images)

    imgs_mask = np.ndarray((total, 650, 650), dtype=np.uint8)
    imgs_order= [None] * total
    i = 0
    print('-'*30)
    print('Creating y training images...')
    print('-'*30)
    for image_name in images:
         # 1. control if the image_id in image_name is contained in df
         # 2. store the mask in the imgs.mask (in the same order of the images)  
        image_id= get_image_id(image_name)
        if len(df[df.ImageId == image_id]) == 0:
            print("ImageId not found on summaryData: {}".format(image_id))
        else:
            img = create_mask_im(df,image_id)
            imgs_mask[i] = img
            imgs_order[i] = image_id
            
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    return imgs_mask, imgs_order



