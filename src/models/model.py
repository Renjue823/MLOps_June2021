# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:33:47 2021

@author: Freja
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_classes, n_neurons = 84):
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
            nn.Linear(in_features=120, out_features=n_neurons),
            nn.Tanh(),
            nn.Linear(in_features=n_neurons, out_features=n_classes),
        )


    def forward(self, x, return_feature=False):
        # if x.ndim != 4:
        #     raise ValueError('Expected input to a 4D tensor')
        # if x.shape[1] != 3 or x.shape[2] != 28 or x.shape[3] != 28:
        #     raise ValueError('Expected each sample to have shape [3, 28, 28]')
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax(logits, dim=1)
        if return_feature:
            return x
        else:
            return probs

