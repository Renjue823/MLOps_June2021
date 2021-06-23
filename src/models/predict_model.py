# change directory 
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)

import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools

from src.models.model import NeuralNetwork
MODEL_PATH = ROOT_PATH + "/src/models/trained_models"
DATA_PATH = ROOT_PATH + "/data/processed"


class Predict:
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        images = torch.load(ROOT_PATH+"/data/processed/val/images.pt")
        #self.predict(images)

    def predict(self, images):
        model = NeuralNetwork(3)
        dict_ = torch.load(MODEL_PATH+"/model_v1.pth")
        model.load_state_dict(dict_)
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)
            predictions = top_class.numpy()
        #self.get_accuracy(predictions)
        return predictions

    def get_accuracy(self,y_hat):
        for i in y_hat:
            for val in i:
                if val != 2:
                    print(val)
        print(len(y_hat))
    
    def evaluate_prediction(self, visualize = True):
        true_labels = torch.load(ROOT_PATH+"/data/processed/val/labels.pt")
        images = torch.load(ROOT_PATH+"/data/processed/val/images.pt")

        predictions = self.predict(images).squeeze()
        if visualize:
            self.confusion_matrix_AF(pred_labels = predictions, true_labels = true_labels, title = 'Confusion matrix for original val data')
        
        TP = sum(true_labels.numpy() == predictions)
        acc = TP/len(true_labels)
        return acc
    
    # Confusion matrix for animal faces
    def confusion_matrix_AF(self, pred_labels, true_labels , title = 'Confusion matrix'):
        classes = ['dog', 'cat', 'wild']
        
        cm = confusion_matrix(pred_labels, true_labels.numpy())
        cmap=plt.cm.Blues
        
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(title +'.jpg')

if __name__ == "__main__":
    images = torch.load(DATA_PATH+"/val/images.pt")
    y = torch.load(DATA_PATH+"/val/labels.pt")
    y_hat = Predict()
    # Predict.get_accuracy(y_hat)

# =============================================================================
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
# test_set = datasets.MNIST('~Freja/MLOps_fork/dtu_mlops/02_code_organisation/CodeOrganisation/data/processed', download=True, train=False, transform=transform)
# test_set = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
#
# images, labels = next(iter(test_set))
# x = Predict(images).predict(images)
# print(x)
# =============================================================================