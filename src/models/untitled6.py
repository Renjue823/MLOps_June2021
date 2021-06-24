import torchdrift
import torch
from sklearn.manifold import Isomap
from matplotlib import pyplot
from torchdrift.detectors.mmd import GaussianKernel , ExpKernel, RationalQuadraticKernel
from tabulate import tabulate
import kornia as K
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
BATCH_SIZE = 64

model = NeuralNetwork(3)
dict_ = torch.load(MODEL_PATH+"/model_v1.pth")
model.load_state_dict(dict_)

feature_extractor = model.feature_extractor

# the input to the drift detector needs to be 1 dim for each img, thus 2 dim for a batch
feature_extractor.final = torch.nn.Flatten()

class datadrifting():
    def __init__(self):
        self.compare_detectors()
        self.compare_detectors(self.blur_tensor)
    
    
    def calculate_score_pvalue_detector(self, kernel = 'gaussian', corruption_function = None):
        if kernel == 'exponential':
            kernel_func = ExpKernel()
        elif kernel == 'rational':
            kernel_func = RationalQuadraticKernel()
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
        
        if len(drift_detector) > 1:
            title_names = ['Gaussian Kernel','Exponential Kernel', 'Rationalquadratic Kernel']
            fig, axs = pyplot.subplots(1,3, figsize = (20,5))
            for i in range(len(drift_detector)):
                N_base = drift_detector[i].base_outputs.size(0)
                mapper = Isomap(n_components=2)
                base_embedded = mapper.fit_transform(drift_detector[i].base_outputs)
                features_embedded = mapper.transform(features[i].detach())
                axs[i].scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
                axs[i].scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
                axs[i].set_title(f'{title_names[i]}: score {score[i]:.2f} p-value.item() {p_val[i]:.2f}')
                return     
        else:
            N_base = drift_detector.base_outputs.size(0)
            mapper = Isomap(n_components=2)
            base_embedded = mapper.fit_transform(drift_detector.base_outputs)
            features_embedded = mapper.transform(features.detach())
            fig, axs = pyplot.subplots()
            axs.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
            axs.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
            axs.set_title(f'score {score:.2f} p-value.item() {p_val:.2f}')
            return 
        
        
    def evaluate_detector(self, kernel = 'gaussian', corruption_function = None):
        drift_detector, features, score, p_val = self.calculate_score_pvalue_detector(kernel)
        #fig, axs = self.visualize_score_pvalue(drift_detector, features, score, p_val)
        
        return score, p_val, drift_detector, features
    
    def compare_detectors(self,corruption_function = None):
        score_gauss,    p_val_gauss   , drift_detector_gauss   , features_gauss    = self.evaluate_detector(kernel = 'gaussian', corruption_function =corruption_function )
        score_exp,      p_val_exp     , drift_detector_exp     , features_exp      = self.evaluate_detector(kernel = 'exponential', corruption_function =corruption_function )
        score_rational, p_val_rational, drift_detector_rational, features_rational = self.evaluate_detector(kernel = 'rational', corruption_function =corruption_function )
        table = [['Kernel', 'score', 'p-value'],
                 ['Gaussian', score_gauss.item(), p_val_gauss.item()],
                 ['Exponential',score_exp.item(), p_val_exp.item()],
                 ['RationalQuadratic',score_rational.item(), p_val_rational.item()]]
        print(tabulate(table))
        drift_detectors = [drift_detector_gauss, drift_detector_exp, drift_detector_rational]
        featuress = [features_gauss, features_exp, features_rational]
        scores = [score_gauss, score_exp, score_rational]
        p_vals = [p_val_gauss, p_val_exp, p_val_rational]
        self.visualize_score_pvalue(drift_detectors, featuress, scores, p_vals)
    
    def blur_tensor(self, img_tensor, kernel_size = 5, sigma = (5,5)):
        kernel_sizes = (kernel_size, kernel_size)
        return K.filters.gaussian_blur2d(img_tensor, kernel_sizes, sigma)
    
    

#%%



datadrifting()

#datadrifting().evaluate_detector()

