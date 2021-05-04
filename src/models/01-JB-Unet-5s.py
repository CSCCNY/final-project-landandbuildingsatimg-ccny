import numpy as np 
#import os
#import skimage.io as io
#import skimage.transform as trans
#import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler ,EarlyStopping
from keras import backend as keras
from keras.metrics import *
from keras.losses import *
from keras import models
#from tensorflow.keras import layers
from sklearn.metrics import classification_report

#custom
import tensorflow as tf

#from new tutorial
#from tensorflow import keras
import numpy as np
#from tensorflow.keras.preprocessing.image import load_img

'''import random
import cv2
import tifffile as tiff

from PIL import Image ,ImageOps'''

#import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator

import math
from datetime import datetime

##beginning of others
from tensorflow.keras.utils import Sequence , get_file

import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, LearningRateScheduler ,EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras

%env SM_FRAMEWORK=tf.keras
import segmentation_models as sm
sm.set_framework('tf.keras')

'''os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= "32"
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(2)'''


def get_model(img_size = (512,512,3) ,num_classes=1 ,filter_sizes=[64, 128, 256]):
    #inputs = keras.Input(shape=img_size + (3,))
    inputs = Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in filter_sizes:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    filter_sizes.reverse()
    filter_sizes.append(32)
    
    for filters in filter_sizes:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    if num_classes==1:
        outputs = layers.Conv2D(filters=num_classes, kernel_size=3, activation="sigmoid", padding="same")(x)
    else:
        outputs = layers.Conv2D(filters=num_classes, kernel_size=3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
#model = get_model((650,650,3), num_classes)
#changes this to load only most recent h5 file, not just _e05.h5
#model_path = "/mnt/hgfs/VMsharedFolder/git/misc/model_checkpoint_unet3_e05.h5"
model_path = "/home/hgamarro/DeepLearning/JB_space/models/Unet/model_checkpoint_unet3_e03_base.h5"
if os.path.isfile(model_path) & True:
    unet = keras.models.load_model(model_path)
    unet.summary()
    print("loaded model from file")
else:
    unet = get_model(num_classes=1)
    unet.summary()
    print("loaded model from code")
    
    
class DataGenerator(Sequence):
    def __init__(self, list_IDs,label_map , img_dir ,mode):
        'Initialization'
        self.list_IDs = list_IDs
        self.label_map = image_label_map
        self.on_epoch_end()
        self.img_dir = img_dir + "/images"
        self.mask_dir = img_dir + "/masks"
        self.mode = mode

    def __len__(self):
        return int(len(self.list_IDs))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index:(index+1)]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y    
    
    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        if self.mode == "train":
            # Generate data
            X, y = self.load_file(list_IDs_temp)
            return X, y
        elif self.mode == "val":
            X, y = self.load_file(list_IDs_temp)
            return X, y        
        
    def load_file(self, id_list):
        list_IDs_temp = id_list
        for ID in list_IDs_temp:
            x_file_path = os.path.join(self.img_dir, ID)
            y_file_path = os.path.join(self.mask_dir, self.label_map.get(ID))
            # Store sample
            X = np.load(x_file_path)
            # Store class
            y = np.load(y_file_path).astype('float32')
        return X, y    
    
out_train_data_dir = '/home/hgamarro/DeepLearning/HG_space/data/processed/Vegas/train'
out_val_data_dir = '/home/hgamarro/DeepLearning/HG_space/data/processed/Vegas/val'

# ====================
# train set
# ====================
all_files = [s for s in os.listdir(out_train_data_dir + "/images/") if s.endswith('.npy')]
all_files.append([s for s in os.listdir(out_train_data_dir + "/masks/") if s.endswith('.npy')] )

image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_files)))}
partition = [item for item in all_files if "image_file" in item]

# ====================
# validation set
# ====================
all_val_files = [s for s in os.listdir(out_val_data_dir + "/images/") if s.endswith('.npy')]
all_val_files.append([s for s in os.listdir(out_val_data_dir + "/masks/") if s.endswith('.npy')] )
val_image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_val_files)))
}
val_partition = [item for item in all_val_files if "image_file" in item]

train_generator = DataGenerator(partition,image_label_map,out_train_data_dir, "train")
val_generator= DataGenerator(val_partition,val_image_label_map,out_val_data_dir, "val")

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 15

n_classes = 1  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
unet.compile(optim, total_loss, metrics)

# define callbacks for learning rate scheduling and best checkpoints saving

model_andor_weight_path = "/home/hgamarro/DeepLearning/JB_space/models/Unet/"
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath = model_andor_weight_path+'best_weights_5.h5'
                                    ,save_freq = 'epoch'
                                    ,verbose = 1
                                    ,save_weights_only=True
                                    ,save_best_only=True
                                    ,mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

earlystop = EarlyStopping(monitor='accuracy'
                         ,min_delta = .01
                         ,patience=3)

start = datetime.now()
print("start: " ,start)
 

history = unet.fit(
    train_generator, 
    steps_per_epoch=len(train_generator), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_generator, 
    validation_steps=len(val_generator),
    use_multiprocessing=True
)

end = datetime.now()
print("end: " ,end)
print("\nTime Taken for testing: %s" % (end-start))

model.save(model_andor_weight_path+"model_unet5.h5")