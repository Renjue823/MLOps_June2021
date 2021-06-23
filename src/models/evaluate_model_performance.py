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

acc_aug = Augmentation().evaluate_prediction( method = 'geometry')
acc_orig = Predict().evaluate_prediction()

acc_noise = Augmentation().evaluate_prediction(method = 'noise', std = 1, mean = 0)

#%%
import numpy as np 
acc_noise = []
std1 = np.arange(0.001,0.1,0.009)
std2 = np.arange(0.1,1,0.05)
std3 = np.arange(1, 2, 0.5)

stds = np.concatenate((std1,std2,std3))
for i in stds:
    tmp = Augmentation().evaluate_prediction(visualize = False, method = 'noise', std = i, mean = 0)
    acc_noise.append(tmp)    
    

#%%

import matplotlib.pyplot as plt 
fig = plt.figure()
p = plt.plot(stds, acc_noise)
plt.hlines(acc_orig, 0, max(stds), 'r', '--')
plt.hlines(acc_aug, 0, max(stds), 'g', '--')
plt.legend(['noisy in data',
            'acc for original data', 
            'acc for geometric augmented'], loc = 7)
plt.xlim([min(stds), max(stds)])
plt.title('performance(accuracy) of the model on noise data')
plt.xlabel('std for gaussian noise')
plt.ylabel('accuracy')
