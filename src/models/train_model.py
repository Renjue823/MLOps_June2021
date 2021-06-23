import os
import torch
from torch import nn, optim

import numpy as np
import pathlib
import matplotlib.pyplot as plt

os.chdir(pathlib.Path().absolute())

from model import NeuralNetwork
from load_data import Train_loader, Test_loader
import pathlib

ROOT_PATH = 'C:/Users/Freja/MLOps_fork/dtu_mlops/MLOps_June2021' 
MODEL_PATH = ROOT_PATH + "/src/models"
DATA_PATH = ROOT_PATH + "/data/processed"


class Train(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """
    def __init__(self):
        self.train()

    def train(self):
        print("training")
        train_loader = Train_loader(64)
        model = NeuralNetwork(n_classes=3)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)       
        epochs = 20

        steps = 0
        train_losses = []
        train_accuracy = 0
        plot_loss = []
        for e in range(epochs):
            running_loss = 0

            for images, labels in train_loader:
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

                train_losses.append(loss.item() / 64)
                train_accuracy += torch.mean(train_equals.type(torch.FloatTensor))

            plot_loss.append(running_loss / len(train_loader))
            print(f"Training loss: {running_loss/len(train_loader)}")
        test_loader = Test_loader(64)
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum()
        
            total += len(labels)
        accuracy = correct / total
        print(accuracy)

        plt.plot(np.arange(epochs), plot_loss)
        plt.savefig(ROOT_PATH + "/reports/figures/loss.png")
        torch.save(model.state_dict(), MODEL_PATH + "/trained_models/model_v1.pth")
    

if __name__ == "__main__":
    Train()
