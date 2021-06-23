import torchdrift
import torch
from sklearn.manifold import Isomap
from matplotlib import pyplot
from torchdrift.detectors.mmd import GaussianKernel , ExpKernel, RationalQuadraticKernel
from tabulate import tabulate

# change directory 
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)

from src.models.load_data import Train_loader
from src.models.model import NeuralNetwork
MODEL_PATH = ROOT_PATH + "/src/models/trained_models"
DATA_PATH = ROOT_PATH + "/data/processed"
BATCH_SIZE = 10

model = NeuralNetwork(3)
dict_ = torch.load(MODEL_PATH+"/model_v1.pth")
model.load_state_dict(dict_)

feature_extractor = model.feature_extractor

# the input to the drift detector needs to be 1 dim for each img, thus 2 dim for a batch
feature_extractor.final = torch.nn.Flatten()

class datadrifting():
    def __init__(self):
        self.compare_detectors()
        self.compare_detectors(self.corruption_function)
    
    
    def calculate_score_pvalue_detector(self, kernel = 'gaussian', corruption_function = None):
        if kernel == 'exponential':
            kernel_func = ExpKernel()
        elif kernel == 'rational':
            kernel_func == RationalQuadraticKernel()
        else:
            kernel_func = GaussianKernel()
        
            
            
        drift_detector = torchdrift.detectors.KernelMMDDriftDetector(kernel = kernel_func)
        torchdrift.utils.fit(Train_loader(BATCH_SIZE), feature_extractor, drift_detector)
        # define the inputs as a test
        inputs, _ = next(iter(Train_loader(BATCH_SIZE)))
        
        # transform using our model 
        if corruption_function is not None:
            inputs = corruption_function(inputs)
     
        features = feature_extractor(inputs)
        # calculate the scores and corresponding p-values
        score = drift_detector(features)
        p_val = drift_detector.compute_p_value(features)
        return drift_detector, features, score, p_val
    
    
    def visualize_score_pvalue(self,drift_detector, features, score, p_val):
        
        N_base = drift_detector.base_outputs.size(0)
        mapper = Isomap(n_components=2)
        base_embedded = mapper.fit_transform(drift_detector.base_outputs)
        features_embedded = mapper.transform(features.detach())
        fig, axs = pyplot.subplots()
        axs.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
        axs.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
        axs.set_title(f'score {score:.2f} p-value.item() {p_val:.2f}')
        return axs
        
    def evaluate_detector(self, kernel = 'gaussian', corruption_function = None):
        drift_detector, features, score, p_val = self.calculate_score_pvalue_detector(kernel)
        axs = self.visualize_score_pvalue(drift_detector, features, score, p_val)
        
        return score, p_val, axs
    
    def compare_detectors(self,corruption_function = None):
        score_gauss,    p_val_gauss   , axs_gauss    = self.evaluate_detector(kernel = 'gaussian', corruption_function =corruption_function )
        score_exp,      p_val_exp     , axs_exp      = self.evaluate_detector(kernel = 'exponential', corruption_function =corruption_function )
        score_rational, p_val_rational, axs_rational = self.evaluate_detector(kernel = 'rational', corruption_function =corruption_function )
        table = [['Kernel', 'score', 'p-value'],
                 ['Gaussian', score_gauss.item(), p_val_gauss.item()],
                 ['Exponential',score_exp.item(), p_val_exp.item()],
                 ['RationalQuadratic',score_rational.item(), p_val_rational.item()]]
        print(tabulate(table))
    def corruption_function(self, x: torch.Tensor):
        return torchdrift.data.functional.gaussian_blur(x, severity=2)

#%%



datadrifting()


