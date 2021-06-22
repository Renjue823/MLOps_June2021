# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:33:47 2021

@author: Freja
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_classes):
        super(NeuralNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=84, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=84, out_channels=120, kernel_size=4, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x, return_feature=False):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax(logits, dim=1)
        if return_feature:
            return x
        else:
            return probs


# =============================================================================
# class NeuralNetwork(nn.Module):
#     def __init__(self, n_classes):
#         super(NeuralNetwork, self).__init__()
# 
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
#             nn.Tanh(),
#         )
# 
#         self.classifier = nn.Sequential(
#             #nn.Linear(in_features=120*4*4, out_features=120),
#             #nn.Tanh(),
#             nn.Linear(in_features=120, out_features=84),
#             nn.Tanh(),
#             nn.Linear(in_features=84, out_features=n_classes),
#         )
# 
# 
#     def forward(self, x, return_feature=False):
#         x = self.feature_extractor(x)
#         print(x.shape)
#         x = torch.flatten(x, 1)
#         logits = self.classifier(x)
#         probs = F.log_softmax(logits, dim=1)
#         if return_feature:
#             return x
#         else:
#             return probs
# =============================================================================
        
        


# =============================================================================
# class NeuralNetwork(nn.Module):
#     def __init__(self, n_classes):
#         super(NeuralNetwork, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             
#             nn.Conv2d(in_channels=3, out_channels=112, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=3),
#             nn.Conv2d(in_channels=112, out_channels=72, kernel_size=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=3),
#             nn.Conv2d(in_channels=72, out_channels=62, kernel_size=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(62),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.Conv2d(in_channels=62, out_channels=32, kernel_size=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2,stride=3),
#             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, padding=0),
#             nn.ReLU()
#             #nn.MaxPool2d(kernel_size=2,stride=2),
#             )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=16, out_features=4096),
#             nn.ReLU(),
#             nn.Dropout2d(0.5),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(),
#             nn.Dropout2d(0.5),
#             nn.Linear(in_features=4096, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=3)
#             )
# 
# 
# 
#     def forward(self, x, return_feature=False):
#         #print('start',x.shape)
#         x = F.pad(x, (3,3,3,3), "constant", 0)
#         #print(x.shape)
#         x = self.feature_extractor(x)
#         #print(x.shape)
#         x = torch.flatten(x, 1)
#         #print('flatten',x.shape)
#         x = self.classifier(x)
#         #print(x.shape)
#         probs = F.log_softmax(x, dim=1)
#         #print('end', probs.shape)
#         if return_feature:
#             return x
#         else:
#             return probs
# 
# =============================================================================
