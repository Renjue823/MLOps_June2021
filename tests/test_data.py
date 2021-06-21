import numpy as np  # to calculate number of detected labels
import pytest
import torch  # to get dataloader
from torchvision import datasets, transforms
import os


ROOT_PATH = "~/Documents/MLOps/MLOps_June2021"
DATA_PATH = ROOT_PATH + "/data/processed"

if os.path.isdir(DATA_PATH):
    check_data = True
else:
    check_data = False

if check_data: 
    from src.models.load_data import train_loader, test_loader, train_data, test_data

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    def test_length():
        """
        Testing if the length of the train and test set are correct
        """
        assert len(train_data) == 14630 and len(test_data) == 1500


    @pytest.mark.parametrize("dataset", [train_loader, test_loader])
    def test_dataset_shape(dataset):
        for images, _ in dataset:
            for image in images:
                s = image[0].size() # to ommit rgb 
                assert s == torch.Size([28, 28]) or s == [728]

    @pytest.mark.parametrize("dataset", [train_loader, test_loader])
    def test_dataset_rgb(dataset):
        for images, _ in dataset:
            for image in images:
                s = len(image)
                assert s == 3



    @pytest.mark.parametrize("dataset", [train_loader, test_loader])
    def test_label_detect(dataset):
        """
        Checks if all labels are present in current part of data set
        """
        label_check = []
        no_labels = 3
        for i in range(0, no_labels):
            label_check.append(False)

        label_check = np.array(label_check)
        for _, labels in dataset:
            for label in labels: 
                if label_check[label] == False:
                    label_check[label] = True
                elif label_check.sum() == no_labels:
                    break
            
            if label_check.sum() == no_labels:
                break

        assert label_check.sum() == no_labels
else:
    print("No data available")