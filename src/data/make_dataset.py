# -*- coding: utf-8 -*-
import logging
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# define root path to the repository on your own computer
ROOT_PATH = 'C:/Users/Freja/MLOps_fork/dtu_mlops/MLOps_June2021'
DATA_PATH_TRAIN = 'data/raw/train/'
DATA_PATH_VAL = 'data/raw/val/'
DATA_PATH_PROC = 'data/processed/'
def main(input_filepath=ROOT_PATH+'/data/raw', output_filepath=ROOT_PATH+'/data/processed'):

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


    # Define a transform to normalize the data
    transformer = transforms.Compose([transforms.Resize(28*2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, ), (0.5, ))])
    
    # define the two dataloaders for the training and validation set
    dataloader_train = ImageFolder(root=DATA_PATH_TRAIN, transform=transformer)
    dataloader_val = ImageFolder(root=DATA_PATH_VAL, transform=transformer)

    # save the training set and corresponding labels to tensors in processed directory
    images = []
    labels = []
    for image,label in dataloader_train:
        images.append(image)
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    torch.save(images, output_filepath+'/train/images.pt')
    torch.save(labels, output_filepath+'/train/labels.pt')

    # save the trvalidation set and corresponding labels to tensors in processed directory
    images = []
    labels = []
    for image,label in dataloader_val:
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
