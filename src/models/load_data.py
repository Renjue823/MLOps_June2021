import sys
# sys.path.append('/Users/lee/Downloads/Renjue/MLOps_June2021')
import torch
import matplotlib.pyplot as plt

# define root path to the repository on your own computer
ROOT_PATH = '/Users/lee/Downloads/Renjue/MLOps_June2021'
DATA_PATH = ROOT_PATH + "/data/processed"

BATCH_SIZE = 64

# Load train data
x = torch.load(DATA_PATH+"/train/images.pt")
y = torch.load(DATA_PATH+"/train/labels.pt")
train_data = []
for i in range(len(x)):
    train_data.append([x[i], y[i]])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Load test data
x = torch.load(DATA_PATH+"/val/images.pt")
y = torch.load(DATA_PATH+"/val/labels.pt")
test_data = []
for i in range(len(x)):
    test_data.append([x[i], y[i]])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# show an example of resized image and its label
# image, label = iter(train_loader).next()
# plt.imshow(image[0].numpy().squeeze(), cmap='Greys_r')
# plt.show()
# print(label[0])

# label: 0 = cat, 1 = dog, 2 = wild

# function to acquire train_loader and test_loader
def loader():
    return train_loader, test_loader


#show an example of resized image and its label
# image, label = iter(train_loader).next()
# plt.imshow(image[0].permute(1, 2, 0))
# plt.show()
# #plt.savefig(fig, '01.jpg')
# print(label[0])
