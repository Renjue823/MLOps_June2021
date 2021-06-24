# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:47:10 2021

@author: annas
"""
import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)

DATA_PATH_PROC = os.environ.get("DATA_PATH_PROC")

import torch
import torch.nn as nn

#from src.models.load_data import Test_loader
from src.models.predict_model import Predict
import kornia as K
from skimage.util import random_noise
import matplotlib.pyplot as plt
import numpy as np
import itertools

class Augmentation(nn.Module):
    def __init__(self):
        super(Augmentation, self).__init__()

        self.augmentation = nn.Sequential(
                                   K.augmentation.RandomAffine(360),
                                   #K.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3)   
                                )
        
        #self.Augment_images()
        
        

    def forward(self, images, method = 'noise', std = 1, mean = 0):
        if method == 'noise':
            new_tensor = self.noise_model(images, std,mean)
        elif method == 'geometry': 
            new_tensor = self.augmentation(images)
        return new_tensor 
    
    def saturate_tensor(self, img_tensor, saturation_factor = 0.1):
        return K.adjust_saturation(img_tensor, saturation_factor)
    
    def brightness_tensor(self, img_tensor, brightness_factor = 0.5):
        return K.enhance.adjust_brightness(img_tensor, brightness_factor)
    
    def blur_tensor(self, img_tensor, kernel_size = 5, sigma = (5,5)):
        kernel_sizes = (kernel_size, kernel_size)
        return K.filters.gaussian_blur2d(img_tensor, kernel_sizes, sigma)
    
    def noise_model(self, tensor_image, std = 0.2, mean = 0):
        noisy_tensor = torch.tensor(random_noise(tensor_image, mode='gaussian', mean=mean, var=std*std, clip=True), dtype = torch.float32)
        return noisy_tensor
    
    def Augment_images(self, method = 'blur', kernel_size = 5):
        images = torch.load(ROOT_PATH +"/data/processed/val/images.pt")
        labels = torch.load(ROOT_PATH +"/data/processed/val/labels.pt")
        
        if method == 'noise':
            new_tensor = self.noise_model(images)#, std, mean)
        elif method == 'geometry': 
            new_tensor = self.augmentation(images)
        elif method == 'blur':
            new_tensor = self.blur_tensor(images, kernel_size)
        elif method == 'brightness':
            new_tensor = self.brightness_tensor(images)
        elif method == 'saturate':
            new_tensor = self.saturate_tensor(images)
        elif len(method) == 2:
            image = images[0]
            new_tensor_noise = self.noise_model(image)#, std, mean)
            new_tensor_aug = self.augmentation(image).squeeze()
            return image, new_tensor_noise, new_tensor_aug
        elif len(method) == 3:
            image = images[0]
            new_tensor_blur = self.blur_tensor(image.unsqueeze(0))
            new_tensor_brightness = self.brightness_tensor(image.unsqueeze(0))
            new_tensor_saturate = self.saturate_tensor(image.unsqueeze(0))
            return image, new_tensor_blur, new_tensor_brightness, new_tensor_saturate
        
        
        torch.save(new_tensor,ROOT_PATH +'/data/processed/val/'+ method + '_images.pt')
        torch.save(labels, ROOT_PATH + '/data/processed/val/'+ method + '_labels.pt')
        return new_tensor

    def predict_augmented_images(self, method):
        #we assume the has already been augmented
        aug_images = torch.load(ROOT_PATH +'/data/processed/val/'+ method + '_images.pt')
        y_hat = Predict().predict(aug_images)
        y_hat = y_hat.squeeze()
        return y_hat
    
    def evaluate_prediction(self, visualize = True, method = 'blur', kernel_size = 5):
        
        self.Augment_images(method, kernel_size)
        
        predictions_aug = self.predict_augmented_images(method)
        
        true_labels = torch.load(DATA_PATH_PROC+"/val/labels.pt")
        
        if visualize:
            Predict().confusion_matrix_AF(pred_labels = predictions_aug,
                                          true_labels = true_labels,
                                          title = 'Confusion matrix for ' + method+ ' data')
        TP = sum(true_labels.numpy() == predictions_aug)
        acc = TP/len(true_labels)
        return acc
    
