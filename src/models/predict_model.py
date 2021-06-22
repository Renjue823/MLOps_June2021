
import sys
import argparse
import os

import torch

# from torch import nn, optim
# import torchvision
from torchvision import datasets, transforms

# import torch.nn.functional as F
import pathlib

os.chdir(pathlib.Path().absolute())


from model import NeuralNetwork

# import numpy as np
import pathlib

# import matplotlib.pyplot as plt
# import plotext.plot as plx


ROOT_PATH = 'C:/Users/Laura/Documents/MLOps/MLOps_June2021' 
MODEL_PATH = ROOT_PATH + "/src/models/trained_models"
DATA_PATH = ROOT_PATH + "/data/processed"


class Predict:
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self, images):
        self.predict(images)

    def predict(self, images):
        model = NeuralNetwork(3)
        dict_ = torch.load(MODEL_PATH+"/model_v1.pth")
        model.load_state_dict(dict_)
        predictions = []
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)
            predictions.append(top_class.numpy())
        self.get_accuracy(predictions)
        return predictions

    def get_accuracy(self,y_hat):
        for i in y_hat:
            for val in i:
                if val != 2:
                    print(val)
        print(len(y_hat))


if __name__ == "__main__":
    images = torch.load(DATA_PATH+"/val/images.pt")
    y = torch.load(DATA_PATH+"/val/labels.pt")
    y_hat = Predict(images)
    # Predict.get_accuracy(y_hat)

# =============================================================================
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
# test_set = datasets.MNIST('~Freja/MLOps_fork/dtu_mlops/02_code_organisation/CodeOrganisation/data/processed', download=True, train=False, transform=transform)
# test_set = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
#
# images, labels = next(iter(test_set))
# x = Predict(images).predict(images)
# print(x)
# =============================================================================
