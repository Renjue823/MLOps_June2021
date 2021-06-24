# import sys
# sys.path.append('/Users/lee/Downloads/Renjue/MLOps_June2021')
import torch
# import helper

# define root path to the repository on your own computer
ROOT_PATH = 'C:/Users/Laura/Documents/MLOps/MLOps_June2021'
DATA_PATH = ROOT_PATH + "/data/processed"


def load_train(batch_size = 64):
    # Load train data
    x = torch.load(DATA_PATH+"/train/images.pt")
    y = torch.load(DATA_PATH+"/train/labels.pt")
    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], y[i]])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test(batch_size = 64):
    # Load test data
    x = torch.load(DATA_PATH+"/val/images.pt")
    y = torch.load(DATA_PATH+"/val/labels.pt")
    test_data = []
    for i in range(len(x)):
        test_data.append([x[i], y[i]])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

# def print_image():
#     train_loader = load_train()
#     image, label = next(iter(train_loader))
#     torch


# if __name__ == '__main__':
#     print_image()


# show an example of resized image and its label
# image, label = iter(train_loader).next()
# plt.imshow(image[0].numpy().squeeze(), cmap='Greys_r')
# plt.show()
# print(label[0])

# label: 0 = cat, 1 = dog, 2 = wild