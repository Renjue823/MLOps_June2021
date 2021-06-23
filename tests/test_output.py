# pytest for model deployment
# Check that output of the scripted model corresponds to output of the non-scripted model
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
sys.path.append(ROOT_PATH)
import pytest
import torch
import torchvision
from torchvision import models

from src.models.model import NeuralNetwork # import the untrained neural network model
from src.models.load_data import loader 

train_loader, test_loader = loader()

MODEL_PATH = ROOT_PATH + '/src/models'
MODEL_DIC_PATH = ROOT_PATH + '/src/models/trained_models'


# load the dictionary of model weights
model_dic = torch.load(MODEL_DIC_PATH+'/model_v1.pth')
model = NeuralNetwork(n_classes=3)
model.load_state_dict(model_dic)

unscripted_model = model
#scripted_model = torch.load(ROOT_PATH+'/model_store/deployable_model.pt')
scripted_model = torch.jit.script(model)

input = torch.randn(1,3,28,28)
unscripted_output = unscripted_model(input)
scripted_output = scripted_model(input)

# torch.topk will return two things, the second one is tensor of indices
# our model has only three possible outputs, we only test the top 1 output
_, unscripted_indices = torch.topk(unscripted_output, k=1)
_, scripted_indices = torch.topk(scripted_output, k=1)

def test_output():
    assert torch.allclose(unscripted_indices, scripted_indices)

