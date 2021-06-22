import sys
import argparse
import os

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from src.models.load_data import train_loader


from src.models.model import NeuralNetwork

import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pdb
# pdb.set_trace()
# 
writer = SummaryWriter()
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)


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

        model = NeuralNetwork(n_classes=3)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        
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

                writer.add_scalar("Loss/train", train_losses[-1], steps)
                writer.add_scalar("Accuracy/train", train_accuracy / steps, steps)

            plot_loss.append(running_loss / len(train_loader))
            print(f"Training loss: {running_loss/len(train_loader)}")

        plt.plot(np.arange(epochs), plot_loss)
        plt.savefig(ROOT_PATH + "/reports/figures/loss.png")
        save_results_to = ROOT_PATH + "/reports/figures/"
        torch.save(model.state_dict(), MODEL_PATH + "/trained_models/model_v1.pth")


if __name__ == "__main__":
    Train()