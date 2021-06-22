# -*- coding: utf-8 -*-
import click
import logging
import pathlib
from dotenv import find_dotenv, load_dotenv
import os
import torch
from torchvision import transforms
from itertools import chain
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.datasets import ImageFolder

# define root path to the repository on your own computer
ROOT_PATH = 'C:/Users/Freja/MLOps_fork/dtu_mlops/MLOps_June2021'  

# define root path to the repository on your own computer
#ROOT_PATH = '/Users/lee/Downloads/Renjue/MLOps_June2021' 
#ROOT_PATH = 'C:/Users/Freja/MLOps_fork/dtu_mlops/MLOps_June2021/'
# ROOT_PATH = ""
DATA_PATH_TRAIN = 'data/raw/train/'
DATA_PATH_VAL = 'data/raw/val/'
DATA_PATH_PROC = 'data/processed/'
#@click.command()
#@click.argument('input_filepath', default=ROOT_PATH + '/data/raw', type=click.Path(exists=True))
#@click.argument('output_filepath', default=ROOT_PATH + '/data/processed', type=click.Path())
def main(input_filepath=ROOT_PATH+'/data/raw', output_filepath=ROOT_PATH+'/data/processed'):

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


    # Define a transform to normalize the data
    transformer = transforms.Compose([transforms.Resize(28*2),
                                    #   transforms.Grayscale(1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, ), (0.5, ))])
    # define the two dataloaders for the training and validation set 
    
    dataloader_train = ImageFolder(root=DATA_PATH_TRAIN, transform=transformer)
    dataloader_val = ImageFolder(root=DATA_PATH_VAL, transform=transformer)
    
    # save the training set and corresponding labels to tensors in processed directory 
    images = []
    labels = []
    count = 0
    for image,label in dataloader_train:
        if count < 6000:
            #print(y)
            images.append(image)
            labels.append(label)
            count += 1
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    torch.save(images, output_filepath+'/train/images.pt')
    torch.save(labels, output_filepath+'/train/labels.pt')
    
    # save the trvalidation set and corresponding labels to tensors in processed directory 
    images = []
    labels = []
    for image,label in dataloader_val:
        #print(y)
        images.append(image)
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    torch.save(images, output_filepath+'/val/images.pt')
    torch.save(labels, output_filepath+'/val/labels.pt')
        
    return None # train_set, test_set
#%%
os.chdir(ROOT_PATH)
main()
print('-----------------------------')
#print(len(train))

