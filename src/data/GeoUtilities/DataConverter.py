import os, json, shutil
import georaster
from skimage.draw import polygon
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import datetime
import time
from itertools import islice

class SpaceNetDataConverter:
    """
        Directory should be in the following structure:
        root_dir/
            geojsons/ (filled with GeoJSON data for each building) each file should be in .geojson data
                buildings_AOI_<AOI_ID>_<LOCATION>_img<ID>.geojson
            raw_tif/ (filled with raw RBG Pan Sharpened Tif Images)
                RBG-PanSharpen_AOI_<AOI_ID>_<LOCATION>_img<ID>.tif
        save_dir/ (required)
            images/ (optional)
            masks/ (optional)
    """
    
    def __init__(self, root_dir, save_dir, aoi_id, location):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.aoi_id = aoi_id
        self.location = location

        self.image_prefix = "RGB-PanSharpen_AOI_{}_{}_img".format(self.aoi_id, self.location)
        self.mask_prefix = "buildings_AOI_{}_{}_img".format(self.aoi_id, self.location)

        geojson_files = [os.path.basename(x) for x in sorted(glob(os.path.join(root_dir, 'geojson/buildings') +'/buildings*') )] 
        self.ids = [g[g.index('img') + 3: g.index('.')] for g in geojson_files]

        if 'masks' not in os.listdir(save_dir):
            os.mkdir(f"{save_dir}/masks")
            
        if 'images' not in os.listdir(save_dir):
            os.mkdir(f"{save_dir}/images")
        
    def convertAllToInput(self):
        self.convertToInput(0, len(self.ids))
        
    def convertToInput(self, start, end):
        num_blank = 0
        pbar = tqdm(self.ids)

        pbar.set_description("{} samples are blank".format(num_blank))
        for img_id in pbar:
            tiff_file = os.path.join(self.root_dir, "RGB-PanSharpen", self.image_prefix + img_id + ".tif")
            tiff = georaster.MultiBandRaster(tiff_file)

            geojson_file = os.path.join(self.root_dir, "geojson/buildings", self.mask_prefix + img_id + ".geojson")
            with open(geojson_file) as gf:
                geojson = json.load(gf)

            image = tiff.r / 2048.0
            mask = self.geoJsonToMask(geojson, tiff)
            if mask is None:
                num_blank += 1
                pbar.set_description("{} samples are blank".format(num_blank))
                continue

            mask = mask.astype(np.uint8)
            image = (image * 255).astype(np.uint8)
            # save images
#             np.save(os.path.join(self.save_dir, "images", img_id), image)
            cv2.imwrite(os.path.join(self.save_dir, "images", img_id+ ".png"), image)
            # save masks
#             np.save(os.path.join(self.save_dir, "masks", img_id + "_mask"), mask)
            cv2.imwrite(os.path.join(self.save_dir, "masks", img_id + ".png"), mask)

        print("Finished!")

    def geoJsonToMask(self, geojson, tiff):
        polyMasks = np.zeros((650,650))
        for i,bldg in enumerate(geojson['features']):
            feature_type = bldg['geometry']['type']
            if 'Polygon' not in feature_type:
                continue
                
            polygons = [bldg['geometry']['coordinates']] if feature_type == "Polygon" else bldg['geometry']['coordinates']

            for mask in polygons:
                rasteredPolygon = np.array(mask[0])
                xs, ys = tiff.coord_to_px(rasteredPolygon[:,0], rasteredPolygon[:,1], latlon=True)

                cc, rr = polygon(xs, ys)
                polyMasks[rr, cc] = 1


        if len(geojson['features']) > 0:
            assert np.max(polyMasks) == 1 and np.min(polyMasks) == 0
            if np.sum(polyMasks) <= 5:
                return None
        else:
            return None

        return polyMasks

# Please use this function to split SpaceNet Train into Train/Val      
def train_val_split(root_dir, save_dir, train_percent):
    
    if 'masks' not in os.listdir(save_dir):
        os.mkdir(f"{save_dir}/masks")

    if 'images' not in os.listdir(save_dir):
        os.mkdir(f"{save_dir}/images")
    
    all_images = list(os.listdir(os.path.join(root_dir, "images")))
    np.random.shuffle(all_images)

    num_train = int(len(all_images) * train_percent)
    val_images = all_images[num_train:]

    for img in tqdm(val_images):
        shutil.move(os.path.join(root_dir, "images", img), os.path.join(save_dir, "images"))
#         shutil.move(os.path.join(root_dir, "masks", img.replace(".png", "_mask.png")), os.path.join(save_dir, "masks"))
        shutil.move(os.path.join(root_dir, "masks", img), os.path.join(save_dir, "masks"))

    
    
class PrePixer(object):
    def __init__(self, img_dir, dump_dir, mode, chunk_size , img_size):
        self.img_dir = img_dir + "/images"
        self.mask_dir = img_dir + "/masks"
        
        self.out_img_dir = dump_dir + "/images"
        self.out_mask_dir = dump_dir + "/masks"
        
        self.mode = mode
        self.chunk_size = chunk_size
        self.img_size = img_size
        
        self.data_gen_args = dict(rescale=1./255,
                     rotation_range=40,
                     horizontal_flip=True)

        self.image_datagen = ImageDataGenerator(**self.data_gen_args)
        self.mask_datagen = ImageDataGenerator(**self.data_gen_args)

  # crate a list of batches based on the total file and desired batch size
    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def pre_pixer(self):  
        # select the file directory
        _img_dir = self.img_dir 
        _mask_dir = self.mask_dir 

        # list all the image files
        img_list = os.listdir(_img_dir)
        # select the file bese on minimum to have 50-50 proportion
        min_len = min([len(img_list)])
        img_list = [{"file_name":item, "label":0} for item in img_list if ".png" in item][: int(min_len)]
        file_list = img_list
        total_len = len(file_list)

        # create list of list based on batch size parameter
        chunks = list(self.chunk(file_list, self.chunk_size))
        
        for idx, meta_files in enumerate(chunks, 1):
            X_file = []
            y_file = []
            batch_size_i = len(meta_files)
        
            #take each image and convert to numpy array
            for item in meta_files:
                label, file_name = item.get("label"), item.get("file_name")
                img_path_i = os.path.join(_img_dir, file_name)
                mask_path_i = os.path.join(_mask_dir, file_name)

                img_i = load_img(img_path_i, target_size=self.img_size, color_mode="rgb")
                img_i = img_to_array(img_i)
                exp_img = np.expand_dims(img_i, axis=0)
                X_file.append(exp_img)

                mask_i = load_img(mask_path_i, target_size=self.img_size , color_mode="grayscale")
                mask_i = img_to_array(mask_i)
                exp_mask = np.expand_dims(mask_i, axis=0)
                y_file.append(exp_mask)

            # a batch of numpy array
            X_file = np.concatenate(X_file)
            y_file = np.concatenate(y_file)
        
            # for validaton we do not need to augment the file so we will save them directly
            if self.mode == "val":
                x_file_name = "image_file_{}.npy".format(idx)
                y_file_name = "label_file_{}.npy".format(idx)
                np.save(os.path.join(self.out_img_dir, x_file_name), X_file)
                np.save(os.path.join(self.out_mask_dir, y_file_name), y_file)
                print("Done batch for validation {}/{}".format(idx, len(chunks)))

          # for train we will have a augmented file of each image
            elif self.mode == "train":
                image_flow = self.image_datagen.flow(
                    X_file,
                    shuffle=False,
                    seed=12,
                    batch_size=32)

                aug_x_set = []
                for aug_x in image_flow:
                    aug_x_set.append(aug_x)
                    break
                aug_x_set = np.concatenate(aug_x_set)

                mask_flow = self.mask_datagen.flow(
                    y_file,
                    shuffle=False,
                    seed=12,
                    batch_size=32)

                aug_y_set = []
                for aug_y in mask_flow:
                    aug_y_set.append(aug_y)
                    break
                aug_y_set = np.concatenate(aug_y_set)

                x_file_name = "image_file_{}.npy".format(idx)
                y_file_name = "label_file_{}.npy".format(idx)

                np.save(os.path.join(self.out_img_dir, x_file_name), aug_x_set)
                np.save(os.path.join(self.out_mask_dir, y_file_name), aug_y_set)

                print("Done batch for train {}/{}".format(idx, len(chunks)))
            else:
                raise Exception("Only <val> or <train> is available as parameter")
        return print("DB Created")
    
    
if __name__ == "__main__":
    """
    Example Usage:
        converter = SpaceNetDataConverter('/data/SpaceNet/AOI_2_Vegas_Train', '/data/SpaceNet/Vegas/train', 2, "Vegas")
        converter.convertAllToInput()
        train_val_split("/data/SpaceNet/Vegas/train", "/data/SpaceNet/Vegas/val", 0.8)
    """