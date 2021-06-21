import numpy as np
import torch
from torch._C import Size

from src.models.model import NeuralNetwork

def test_model_output():
    model = NeuralNetwork(n_classes=3)
    assert model.forward(torch.zeros((100, 3, 28, 28))).shape == torch.Size((100, 3))

def test_probs_size():
    model = NeuralNetwork(n_classes=3)
    x = torch.zeros((10, 3, 28, 28))
    probs = model.forward(x,return_feature=True)
    s = probs.size()
    assert s == torch.Size([10,120])

def test_probs_values():
    model = NeuralNetwork(n_classes=3)
    x = torch.zeros((10, 3, 28, 28))
    probs = model.forward(x,return_feature=True)
    for single_image_probs in probs:
        for p in single_image_probs:
            assert p > -1 and p < 1
