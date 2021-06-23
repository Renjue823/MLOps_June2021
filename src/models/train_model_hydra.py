# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:27:06 2021

@author: Freja
"""
import os
import torch
from torch import nn, optim
import numpy as np
import pathlib
import matplotlib.pyplot as plt

os.chdir(pathlib.Path().absolute())

from model import NeuralNetwork
from load_data import Train_loader

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

ROOT_PATH = 'C:/Users/Freja/MLOps_fork/dtu_mlops/MLOps_June2021' 
MODEL_PATH = ROOT_PATH + "/src/models"
DATA_PATH = ROOT_PATH + "/data/processed"

def Train(cfg: DictConfig) -> None:
    seed = cfg.parameters.seed
    torch.manual_seed(seed)
    log.info("Start training")

    model = NeuralNetwork(n_classes=3)
    criterion = nn.NLLLoss()
    lr = cfg.parameters.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)       
    epochs = cfg.parameters.epochs
    batch_size = cfg.parameters.batch_size
    train_loader = Train_loader(batch_size)
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

    plt.plot(np.arange(epochs), plot_loss)
    plt.savefig(f"{os.getcwd()}/loss.png")
    log.info("Finish!!")
    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")
    

@hydra.main(config_name="config")
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    Train(cfg)


if __name__ == '__main__':
    run_model()
