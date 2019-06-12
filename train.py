from __future__ import print_function
from .reader import name_file 
from .reader import loading_train_data
from .preprocessing import preprocess_ris
from .preprocessing import normalizing_imgs
from .network import get_unet
from .network import dice_coef_loss
from .network import dice_coef



import json
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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
from sklearn.model_selection import train_test_split


smooth=1
img_rows=96
img_cols=96

def train_mod_old():
    img_rows=96
    img_cols=96
  
    Xtrain,Ytrain = loading_train_data()
    Xtrain=preprocess_ris(Xtrain,img_cols,img_rows)
    Ytrain=preprocess_ris(Ytrain,img_cols,img_rows)
    Xtrain= normalizing_imgs(Xtrain)
    Ytrain= normalizing_imgs(Ytrain)
   
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    # (default values for the arguments of model.fit) 
    # fit(x=None, y=None, batch_size=None, epochs=1, 
    #    verbose=1, callbacks=None, validation_split=0.0, 
    #    validation_data=None, shuffle=True, class_weight=None, 
    #    sample_weight=None, initial_epoch=0, steps_per_epoch=None, 
    #    validation_steps=None, validation_freq=1)
    
    model.fit(Xtrain, Ytrain, batch_size=2, epochs=20, verbose=2, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    return model

def train_mod(Xtrain,Ytrain):
    img_rows=96
    img_cols=96
   
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    # (default values for the arguments of model.fit) 
    # fit(x=None, y=None, batch_size=None, epochs=1, 
    #    verbose=1, callbacks=None, validation_split=0.0, 
    #    validation_data=None, shuffle=True, class_weight=None, 
    #    sample_weight=None, initial_epoch=0, steps_per_epoch=None, 
    #    validation_steps=None, validation_freq=1)
    
    model.fit(Xtrain, Ytrain, batch_size=2, epochs=20, verbose=2, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    
    

    return model
