import tensorflow as tf
import numpy as np
import logging   
import cv2

def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

def unsplit_image(tiles4, image_shape):
    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])

def slice_images(data_generator, mask_generator, directory, slices, img_height, batch_size, batch_n = None):
# 	Slice images into specified # of slices.
#   If batch_n = None slicing will be done in all the batches of the image generator. If batch_n = int slicing will be done in the first batch_n batches.

    #now we will Create and configure logger 
    logging.basicConfig(filename="slice_images.log", 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w') 

    #Let us Create an object 
    logger=logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 

    tile_size = int(img_height/np.sqrt(slices))
    logger.info(f'Resulting image size: {tile_size} x {tile_size}')
    batch_index = 0
    
    if batch_n != None:
        logger.info(f'It will run for {batch_n} batches')
        while batch_index <= batch_n-1:
            x = data_generator.next()
            m = mask_generator.next()
            for i in range(batch_size):
                data_tiles = split_image(x[i,:,:,:], [tile_size, tile_size])
                mask_tiles = split_image(m[i,:,:,:], [tile_size, tile_size])
                for t in range(data_tiles.shape[0]):
                    cv2.imwrite(f'{directory}/Data/Train_Batch_{batch_index}_Image_{i}_Tile_{t}.png', 
                                                          data_tiles[t,:,:,:])
                    cv2.imwrite(f'{directory}/Mask/Mask_Batch_{batch_index}_Image_{i}_Tile_{t}.png', 
                                                          mask_tiles[t,:,:,:])
            logger.info(f'Just finished batch {batch_index}')
            batch_index = batch_index + 1
            
    else:
        logger.info(f'It will run for all batches')
        while batch_index <= data_generator.batch_index:
            x = data_generator.next()
            m = mask_generator.next()
            for i in range(batch_size):
                data_tiles = split_image(x[i,:,:,:], [tile_size, tile_size])
                mask_tiles = split_image(m[i,:,:,:], [tile_size, tile_size])
                for t in range(data_tiles.shape[0]):
                    tf.keras.preprocessing.image.save_img(f'../Cropping_Test/Train_Batch_{batch_index}_Image_{i}_Tile_{t}.png', 
                                                          data_tiles[t,:,:,:], data_format=None, file_format=None, scale=True)
                    tf.keras.preprocessing.image.save_img(f'../Cropping_Test/Mask_Batch_{batch_index}_Image_{i}_Tile_{t}.png', 
                                                          mask_tiles[t,:,:,:], data_format=None, file_format=None, scale=True)
            logger.info(f'Just finished batch {batch_index}')
            batch_index = batch_index + 1
    logger.info(f'Slicing finished')