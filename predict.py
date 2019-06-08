from __future__ import print_function
from .reader import name_file 
from .reader import loading_test_data
from .preprocessing import preprocess_ris
from .preprocessing import normalizing_imgs

import json
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.ion()
from shapely.geometry import Point, Polygon
import shapely.wkt

import geopandas as gpd
from rasterio.plot import show
import sys
import geojson
from descartes import PolygonPatch


from skimage.transform import resize
import skimage.draw
from skimage.io import imsave, imread


from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
plt.rcParams['figure.figsize'] = (8, 8)
from sklearn.model_selection import train_test_split


# Prettier plotting with seaborn
import seaborn as sns; 
sns.set(font_scale=1.5)
sns.set_style("white")


def predict_mod(model):
    img_rows=96
    img_cols=96
  
    Ytest,indices_test = loading_test_data()
    Ytest=preprocess_ris(Ytest,img_cols,img_rows)
    Ytest= normalizing_imgs(Ytest)
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model.load_weights('weights.h5')
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(Ytest, verbose=2)
    np.save('data/imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'data'
    
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, indices_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
    return np.save(os.path.join(pred_dir,'imgs_mask_test.npy'), imgs_mask_test) 


