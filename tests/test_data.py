import numpy as np  # to calculate number of detected labels
import pytest
import torch  # to get dataloader
from torchvision import datasets, transforms

ROOT_PATH =  "~/Documents/MLOps/MLOps_June2021"
DATA_PATH = ROOT_PATH + "/data/processed"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_set = datasets.MNIST(
    DATA_PATH,
    download=True,
    train=True,
    transform=transform,
)
test_set = datasets.MNIST(
    DATA_PATH,
    download=True,
    train=False,
    transform=transform,
)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

# class Helpers:
#     def get_len(data, l=[]):
#         l.append(len(data))
#         if len(data[0]) > 1:
#             Helpers.get_len(data,l)
#         else:
#             return l


def test_length():
    """
    Testing if the length of the train and test set are correct
    """
    assert len(train_set) == 60000 and len(test_set) == 10000


@pytest.mark.parametrize("dataset", [train_set, test_set])
def test_dataset_shape(dataset):
    for datapoint in dataset.data:
        s = datapoint.shape
        assert s == torch.Size([28, 28]) or s == [728]


@pytest.mark.parametrize("dataset", [train_set, test_set])
def test_label_detect(dataset):
    """
    Checks if all labels are present in current part of data set
    """
    label_check = []
    no_labels = 10
    for i in range(0, no_labels):
        label_check.append(False)

    label_check = np.array(label_check)
    for datapoint in dataset:
        label = datapoint[1]
        if label_check[label] == False:
            label_check[label] = True
        elif label_check.sum() == no_labels:
            break

    assert label_check.sum() == no_labels
