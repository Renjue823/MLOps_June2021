# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data.dataset import random_split 

DATA_PATH_TRAIN = 'data/raw/train/'
DATA_PATH_VAL = 'data/raw/val/'


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():#(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

  
    # data preparation found in the project below 
    # https://blog.jovian.ai/apply-deep-neural-network-on-the-animal-faces-dataset-f573cb732e63
    dataset = ImageFolder(DATA_PATH_TRAIN, transform=transforms.ToTensor())
    val_ds = ImageFolder(DATA_PATH_VAL, transform=transforms.ToTensor())

    torch.manual_seed(43)
    test_size = 5000
    train_size = len(dataset) - test_size    
    train_ds, test = random_split(dataset, [train_size, test_size])
    
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())    
    
    main()
