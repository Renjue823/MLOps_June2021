
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
import pathlib
from load_data import load_test

ROOT_PATH = 'C:/Users/Laura/Documents/MLOps/MLOps_June2021' 
MODEL_PATH = ROOT_PATH + "/src/models/trained_models"
DATA_PATH = ROOT_PATH + "/data/processed"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters = {
    'lr': 800*10e-6,
    'batch_size': 20, 
    'epochs': 18, 
    'n_neurons': 50
}

class Predict:
    """Helper class that will help launch class methods as commands
    from a single script
    """

    # def __init__(self, images):
    #     # self.predict(images)

    def predict(self, images):
        

        model = NeuralNetwork(n_classes=3, n_neurons=parameters['n_neurons'])
        dict_ = torch.load(MODEL_PATH+"/model_v1.pth")
        model.load_state_dict(dict_)
        predictions = []
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)
            predictions.append(top_class.numpy())
        # self.get_accuracy(predictions)
        return predictions

    def get_accuracy(self,y_hat):
        for i in y_hat:
            for val in i:
                if val != 2:
                    print(val)
        print(len(y_hat))


if __name__ == "__main__":
    
#     
#     # y_hat = Predict()

# # =============================================================================
# # transform = transforms.Compose([transforms.ToTensor(),
# #                                 transforms.Normalize((0.5,), (0.5,))])
# # test_set = datasets.MNIST('~Freja/MLOps_fork/dtu_mlops/02_code_organisation/CodeOrganisation/data/processed', download=True, train=False, transform=transform)
# # test_set = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
# #
#     images, labels = next(iter(test_loader))
#     x = Predict(images).predict(images)
#     print(x)
# =============================================================================

    test_loader = load_test(batch_size=parameters['batch_size'])
    correct, total = 0,0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = torch.tensor(Predict().predict(images))

        predictions = torch.max(outputs, 1)[1].to(device)
        correct += (outputs == labels).sum()

        total += len(labels)

    accuracy = correct / total 
    print(correct)
    print(accuracy)
