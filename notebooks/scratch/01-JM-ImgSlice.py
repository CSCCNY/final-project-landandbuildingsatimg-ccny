import sys
sys.path
sys.path.append('/home/juanp.montoya/NeuralNetworks/Final_Project/final-project-landandbuildingsatimg-ccny/')

import os # accessing directory structure
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np # linear algebra
import pandas as pd
from src.data.GeoUtilities import ImgSlice


os.chdir('/home/juanp.montoya/NeuralNetworks/Final_Project/final-project-landandbuildingsatimg-ccny/data/raw/input')


nRowsRead = None 
df2 = pd.read_csv('metadata.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'metadata.csv'
nRow, nCol = df2.shape

img_height = 2448
img_width = 2448

data_gen = ImageDataGenerator(rescale = 1.)
data_gen2 = ImageDataGenerator(rescale = 1.)

batch_size = 32

data_generator = data_gen.flow_from_dataframe(dataframe=df2.loc[df2['split'] == 'train'], 
                                            directory="", 
                                            x_col="sat_image_path", 
                                            y_col="mask_path", 
                                            class_mode=None,
                                            seed = 0,
                                            target_size=(img_height,img_width), 
                                            batch_size=batch_size)

mask_generator = data_gen2.flow_from_dataframe(dataframe=df2.loc[df2['split'] == 'train'], 
                                            directory="", 
                                            x_col="mask_path", 
                                            y_col="mask_path", 
                                            class_mode=None,
                                            seed = 0,
                                            target_size=(img_height,img_width), 
                                            batch_size=batch_size)

batch_n = None
slices = 16
directory = '../../interim/Cropped'

ImgSlice.slice_images(data_generator, mask_generator, directory, slices, img_height, batch_size, batch_n)