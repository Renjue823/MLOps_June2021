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

# define root path to the repository on your own computer
ROOT_PATH = '/Users/lee/Downloads/Renjue/MLOps_June2021' 

@click.command()
@click.argument('input_filepath', default=ROOT_PATH + '/data/raw', type=click.Path(exists=True))
@click.argument('output_filepath', default=ROOT_PATH + '/data/processed', type=click.Path())
def main(input_filepath=ROOT_PATH+'/data/raw', output_filepath=ROOT_PATH+'/data/processed'):

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    TRAIN_PATH = input_filepath+'/afhq/train'
    TEST_PATH = input_filepath+'/afhq/val'
    CAT_TRAIN_PATH = TRAIN_PATH+'/cat'
    CAT_TEST_PATH = TEST_PATH+'/cat'
    DOG_TRAIN_PATH = TRAIN_PATH+'/dog'
    DOG_TEST_PATH = TEST_PATH+'/dog'
    WILD_TRAIN_PATH = TRAIN_PATH+'/wild'
    WILD_TEST_PATH = TEST_PATH+'/wild'

    category_names = ['cat', 'dog','wild']
    train_folder_path = [CAT_TRAIN_PATH, DOG_TRAIN_PATH, WILD_TRAIN_PATH]
    test_folder_path = [CAT_TEST_PATH, DOG_TEST_PATH, WILD_TEST_PATH]
    train_images = []
    train_set = []
    test_images = []
    test_set = []

    # access the number of images in each category and acquire images in train_set and test_set
    for folder in train_folder_path:
        train_images.append(len(os.listdir(folder)))
        train_set.append(os.listdir(folder))
    for folder in test_folder_path:
        test_images.append(len(os.listdir(folder)))
        test_set.append(os.listdir(folder))

    # flatten train_set and test_set
    train_set = list(chain.from_iterable(train_set))
    test_set = list(chain.from_iterable(test_set))

    for i in range(len(train_set)):
        if i < train_images[0]:
            train_set[i] = Image.open(train_folder_path[0]+'/'+train_set[i]).convert("L") # converting grey scale 
        elif train_images[0] <= i < train_images[0]+train_images[1]:
            train_set[i] = Image.open(train_folder_path[1]+'/'+train_set[i]).convert("L")
        else:
            train_set[i] = Image.open(train_folder_path[2]+'/'+train_set[i]).convert("L")
    for i in range(len(test_set)):
        if i < test_images[0]:
            test_set[i] = Image.open(test_folder_path[0]+'/'+test_set[i]).convert("L")
        elif test_images[0] <= i < test_images[0]+test_images[1]:
            test_set[i] = Image.open(test_folder_path[1]+'/'+test_set[i]).convert("L")
        else:
            test_set[i] = Image.open(test_folder_path[2]+'/'+test_set[i]).convert("L")

    # resize images
    img_size = 50
    for i in range(len(train_set)):
        train_set[i] = train_set[i].resize((img_size,img_size), Image.ANTIALIAS)
        train_set[i] = np.asarray(train_set[i])/255
    for i in range(len(test_set)):
        test_set[i] = test_set[i].resize((img_size,img_size), Image.ANTIALIAS)
        test_set[i] = np.asarray(test_set[i])/255

    train_label = np.array([])
    test_label = np.array([])
    for id, num in enumerate(train_images):
        train_label = np.concatenate((train_label, np.ones(num)*id), axis=0)
    for id, num in enumerate(test_images):
        test_label = np.concatenate((test_label, np.ones(num)*id), axis=0)
    train_label = train_label.tolist()
    test_label = test_label.tolist()  

    # shape [num of images, 50, 50]
    train_set = np.stack(train_set, axis=0)
    print('The shape of training set: {}'.format(train_set.shape))
    test_set = np.stack(test_set, axis=0)
    print('The shape of test set: {}'.format(test_set.shape))
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, )),
                                    ])

    train_set = transform(train_set)
    test_set = transform(test_set)
    print('One image in training set after transformation: {}'.format(train_set[0]))
    return train_set, test_set

train, test = main()
print('-----------------------------')
print(len(train))



# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

