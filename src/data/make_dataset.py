# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from GeoUtilities import DataConverter

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    
    input_path = str(project_dir) + '/data/raw/AOI_2_Vegas_Train'
    output_path = str(project_dir) + '/data/interim/Vegas/train'
    val_path = str(project_dir) + '/data/interim/Vegas/val'
    
    converter = DataConverter.SpaceNetDataConverter(input_path, output_path, 2, "Vegas")
    converter.convertAllToInput()
    
    DataConverter.train_val_split(output_path, val_path, 0.8)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main(project_dir)
