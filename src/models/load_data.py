import sys
# sys.path.append('/Users/lee/Downloads/Renjue/MLOps_June2021')
import torch
import matplotlib.pyplot as plt

# define root path to the repository on your own computer
import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)
DATA_PATH = ROOT_PATH + "/data/processed"

# Load train data
x = torch.load(DATA_PATH+"/train/images.pt")
y = torch.load(DATA_PATH+"/train/labels.pt")
train_data = []
for i in range(len(x)):
    train_data.append([x[i], y[i]])

def Train_loader(BATCH_SIZE):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

# Load test data
x = torch.load(DATA_PATH+"/val/images.pt")
y = torch.load(DATA_PATH+"/val/labels.pt")
test_data = []
for i in range(len(x)):
    test_data.append([x[i], y[i]])


def Test_loader(BATCH_SIZE):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

# show an example of resized image and its label
# image, label = iter(train_loader).next()
# plt.imshow(image[0].numpy().squeeze(), cmap='Greys_r')
# plt.show()
# print(label[0])


# label: 0 = cat, 1 = dog, 2 = wild

