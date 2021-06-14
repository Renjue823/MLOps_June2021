import sys
sys.path.append('/Users/lee/Downloads/Renjue/MLOps_June2021')
import torch
import matplotlib.pyplot as plt

# define root path to the repository on your own computer
ROOT_PATH = '/Users/lee/Downloads/Renjue/MLOps_June2021' 

# load data loaders
train_loader = torch.load(ROOT_PATH + '/data/processed/train_loader')
test_loader = torch.load(ROOT_PATH + '/data/processed/test_loader')

# show an example of resized image and its label
image, label = iter(train_loader).next()
plt.imshow(image[0].numpy().squeeze(), cmap='Greys_r')
plt.show()
print(label[0])

# label: 0 = cat, 1 = dog, 2 = wild