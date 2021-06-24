import sys
import argparse
import os

#import tensorflow as tf
import torch
from torch import nn, optim
#from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from load_data import load_train, load_test

import io
import PIL
import time


# import tensorflow as tf


# import torch.nn.functional as F
import pathlib

os.chdir(pathlib.Path().absolute())

from model import NeuralNetwork

import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pdb

# writer = SummaryWriter()

# ROOT_PATH = str(pathlib.Path(*pathlib.Path().absolute().parts[:-2]))
ROOT_PATH = 'C:/Users/Laura/Documents/MLOps/MLOps_June2021' 
MODEL_PATH = ROOT_PATH + "/src/models"
DATA_PATH = ROOT_PATH + "/data/processed"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        self.train()

    def train(self):
        print("training")

        parameters ={'lr': 0.0008500000000000001, 'n_neurons': 3, 'epochs': 40, 'batch_size': 41}

        model = NeuralNetwork(n_classes=3, n_neurons=parameters['n_neurons'])
        criterion = nn.NLLLoss()

        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])

        train_loader = load_train(batch_size=parameters['batch_size'])

        # x = torch.load(DATA_PATH+"/train/images.pt")
        # y = torch.load(DATA_PATH+"/train/labels.pt")

        # train_data = []
        # for i in range(len(x)):
        #     train_data.append([x[i], y[i]])

        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        

        epochs = parameters['epochs']  # 20
        steps = 0
        train_losses = []
        train_accuracy = 0
        plot_loss = []
        for e in range(epochs):
            running_loss = 0

            for images, labels in train_loader:
# =============================================================================
#                 f, axarr = plt.subplots(4,1)
#                 print(images[0].shape)
#                 axarr[0].imshow(images[0].permute(1,2,0))
#                 axarr[1].imshow(images[1].permute(1,2,0))
#                 axarr[2].imshow(images[2].permute(1,2,0))
#                 axarr[3].imshow(images[3].permute(1,2,0))
#                 plt.show()
#                 print(labels[:4])
#                 time.sleep(5)
# =============================================================================

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1

                train_ps = torch.exp(model(images))
                train_top_p, train_top_class = train_ps.topk(1, dim=1)
                train_equals = train_top_class == labels.view(*train_top_class.shape)

                train_losses.append(loss.item() / parameters['batch_size'])
                train_accuracy += torch.mean(train_equals.type(torch.FloatTensor))

                #writer.add_scalar("Loss/train", train_losses[-1], steps)
                #writer.add_scalar("Accuracy/train", train_accuracy / steps, steps)

            plot_loss.append(running_loss / len(train_loader))
            print(f"Training loss: {running_loss/len(train_loader)}")

        plt.plot(np.arange(epochs), plot_loss)
        plt.savefig(ROOT_PATH + "/reports/figures/loss.png")
        save_results_to = ROOT_PATH + "/reports/figures/"
        torch.save(model.state_dict(), MODEL_PATH + "/trained_models/model_v1.pth")

        test_loader = load_test(batch_size=parameters['batch_size'])
        correct, total = 0,0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = model(images)
        
            predictions = torch.max(outputs, 1)[1].to(device)
            correct += (predictions == labels).sum()
        
            total += len(labels)
    
        accuracy = correct / total 
        print(accuracy)
        return accuracy
    

if __name__ == "__main__":
    Train()
