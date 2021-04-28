import tensorflow as tf
import numpy as np
import logging 
from src.data.GeoUtilities import ImgSlice
import os

def RGB_mapping_to_class(label):
#     Convert RGB mask to categorical array.
#     urban_land         [0 ,255 ,255]    --> 1
#     agriculture_land   [255 ,255 ,0]    --> 2   
#     rangeland          [255 ,0 ,255]    --> 3
#     forest_land        [0 ,255 ,0]      --> 4
#     water              [0 ,0 ,255]      --> 5
#     barren_land        [255 ,255 ,255]  --> 6
#     unknown            [0 ,0 ,0]        --> 0


    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (255, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (255, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
#   Convert categorical array to RGB image.
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


def mask_to_label(mask_generator, directory, slices, img_height, batch_size, batch_n = None):
# 	Transform RGB mask into labels
#   If batch_n = None slicing will be done in all the batches of the image generator. If batch_n = int slicing will be done in the first batch_n batches.

    #now we will Create and configure logger 
    logging.basicConfig(filename="mask_to_label.log", 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w') 

    #Let us Create an object 
    logger=logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.INFO) 

    logger.info(f'Lets Start \n \n \n')
    tile_size = int(img_height/np.sqrt(slices))
    batch_index = 0
    logger.info(f'{os.getcwd()}')

    if batch_n != None:
        logger.info(f'It will run for {batch_n} batches \n \n \n')
        while batch_index <= batch_n-1:
            m = mask_generator.next()
            for i in range(batch_size):
                mask_tiles = ImgSlice.split_image(m[i,:,:,:], [tile_size, tile_size])
                for t in range(mask_tiles.shape[0]):
                    mask_label = RGB_mapping_to_class(mask_tiles[t,:,:,:])
                    np.save(f'{directory}/Mask_Batch_{batch_index}_Image_{i}_Tile_{t}.npy', mask_label)
                    
            logger.info(f'Just finished batch {batch_index} \n \n \n')
            batch_index = batch_index + 1
            
    else:
        logger.info(f'It will run for all batches \n \n \n')
        while batch_index <= data_generator.batch_index:
            m = mask_generator.next()
            for i in range(batch_size):
                mask_tiles =ImgSlice.split_image(m[i,:,:,:], [tile_size, tile_size])
                for t in range(mask_tiles.shape[0]):
                    mask_label = RGB_mapping_to_class(mask_tiles[t,:,:,:])
                    np.save(f'{directory}/Mask_Batch_{batch_index}_Image_{i}_Tile_{t}.npy', mask_label)
                    
            logger.info(f'Just finished batch {batch_index} \n \n \n')
            batch_index = batch_index + 1
    logger.info(f'Slicing finished')
