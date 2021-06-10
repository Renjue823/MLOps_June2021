import sys
import argparse
import os

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
#import torch.nn.functional as F
import pathlib
os.chdir(pathlib.Path().absolute())




from model import NeuralNetwork

import numpy as np
import pathlib
import matplotlib.pyplot as plt
#import plotext.plot as plx

MODEL_PATH = pathlib.Path().absolute()
ROOT_PATH =  str(pathlib.Path(*MODEL_PATH.parts[:-2]))
MODEL_PATH = str(MODEL_PATH)
DATA_PATH = ROOT_PATH + "/data/processed"


class Train(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.train()
    
    def train(self):
        print('training')
        # Implement training loop 
        model = NeuralNetwork(10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
        # Download and load the data
        train_set = datasets.MNIST('~Laura/Documents/MLOps/git_mlops/data/processed', download=True, train=True, transform=transform)
        train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)
        epochs = 3#20
        steps = 0
        train_losses = []
        plot_loss = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1
                train_losses.append(loss.item()/64)
            plot_loss.append(running_loss/len(train_set))
            print(f"Training loss: {running_loss/len(train_set)}")

        plt.plot(np.arange(epochs),plot_loss)
        plt.savefig(ROOT_PATH+'/reports/figures/loss.png')
        torch.save(model.state_dict(), MODEL_PATH+'/trained_models/model_v1.pth')

if __name__ == '__main__':
    Train()
    
    
    
    
    
    
    