# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:31:43 2021

@author: annas
"""
import torch 

import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)


from src.models.augment_model import Augmentation
from src.models.predict_model import Predict 

import numpy as np 
import matplotlib.pyplot as plt 


acc_orig = Predict().evaluate_prediction()

acc_blur = Augmentation().evaluate_prediction( method = 'blur')

acc_bright = Augmentation().evaluate_prediction(method = 'brightness')

acc_sat = Augmentation().evaluate_prediction(method = 'saturate')

#%%

img, img_blur, img_bright, img_sat  = Augmentation().Augment_images(method = ['blur', 'brightness', 'saturate'])

plt.figure()
plt.imshow(img.permute(1,2,0))
plt.show()
plt.savefig('reports/figures/original_image.png')

plt.imshow(img_blur.squeeze().permute(1,2,0))
plt.show()
plt.savefig('reports/figures/blur_image.png')

plt.imshow(img_bright.squeeze().permute(1,2,0))
plt.show()
plt.savefig('reports/figures/bright_image.png')

plt.imshow(img_sat.squeeze().permute(1,2,0))
plt.show()
plt.savefig('reports/figures/sat_image.png')

#%% create plot for what influence white noise has on 
acc_noise = []
std1 = np.arange(0.001,0.1,0.009)
std2 = np.arange(0.1,1,0.05)
std3 = np.arange(1, 2, 0.5)

stds = np.concatenate((std1,std2,std3))


kernel_sizes = np.arange(1, 20, 2)
tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 1)
acc_noise.append(tmp)    
    
tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 3)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 5)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 7)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 9)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 11)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 13)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 15)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 17)
acc_noise.append(tmp)    

tmp = Augmentation().evaluate_prediction(visualize = False,
                                         method = 'blur',
                                         kernel_size = 19)
acc_noise.append(tmp)    
#%%

fig = plt.figure()
plt.plot(kernel_sizes, acc_noise)
plt.hlines(acc_orig, 1, max(kernel_sizes), 'r', '--')
plt.hlines(acc_bright, 1, max(kernel_sizes), 'g', '--')
plt.hlines(acc_sat, 1, max(kernel_sizes), 'k', '--')
plt.legend(['noisy in data',
            'acc for original data', 
            'acc for brighter data', 
            'acc for saturated data'], loc = 7)
plt.xlim([min(kernel_sizes), max(kernel_sizes)])
plt.title('performance(accuracy) of the model on noise data')
plt.xlabel('std for gaussian noise')
plt.ylabel('accuracy')

plt.savefig('reports/figures/noise_accuracy.jpg')
