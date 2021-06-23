import torch
from torch.autograd import Variable
from torch.serialization import save
import torchvision
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import sys
import matplotlib.pyplot as plt
# the root path is defined under .env
ROOT_PATH = os.environ.get("ROOT_PATH")
sys.path.append(ROOT_PATH)
from src.models.model import NeuralNetwork # import the untrained neural network model
from src.models.load_data import loader 

# load train_loader and test_loader
train_loader, test_loader = loader()

MODEL_PATH = ROOT_PATH + '/src/models'
MODEL_DIC_PATH = ROOT_PATH + '/src/models/trained_models'


# load the dictionary of model weights
model_dic = torch.load(MODEL_DIC_PATH+'/model_v1.pth')
model = NeuralNetwork(n_classes=3)
model.load_state_dict(model_dic)

# test the model
images, labels = iter(test_loader).next()
print('The true label of one image:{}'.format(labels[0]))

example_image = images[0]
image =  images[0].unsqueeze(0) # add one dimension to the image [1, 3, 56, 56]
pre = model(image)
print('The predicted label of one image:{}'.format(pre[0, :]))

# script model with TorchScript
script_model = torch.jit.script(model)
script_model.save(ROOT_PATH+'/model_store/deployable_model.pt')

'''
To enable torch model archiver: The model is deployed as Animal_Classifier.mar
torch-model-archiver --model-name Animal_Classifier --version 1.0 --serialized-file /Users/lee/Downloads/Renjue/MLOps_June2021/model_store/deployable_model.pt --export-path /Users/lee/Downloads/Renjue/MLOps_June2021/model_store --extra-files index_to_name.json --handler image_classifier
'''

'''
Model Deployment
torchserve --start --ncs --model-store /Users/lee/Downloads/Renjue/MLOps_June2021/model_store --models Animal_Classifier=Animal_Classifier.mar
curl http://127.0.0.1:8080/predictions/Animal_Classifier -T cat_1.jpg
(my_image.jpg should be an image stored under model_store directory)
'''
# save one example image as tensor 
torch.save(image, '/Users/lee/Downloads/Renjue/MLOps_June2021/model_store/tensor.pt')
# show the example image
# plt.imshow(example_image.permute(1, 2, 0))
# plt.show()

