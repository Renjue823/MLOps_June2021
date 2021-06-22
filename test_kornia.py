from torch import nn
import torch
from PIL import Image
import numpy as np
import kornia as K 

import logging
import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = os.environ.get("ROOT_PATH")
os.chdir(ROOT_PATH)


#%%
class PreProcess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self) -> None:
        super().__init__()
 
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float()
    
#%%
def imshow(input: torch.Tensor):
#create grid having 2 rows of images
    output: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, 
    padding=5)
#Convert tensor images to numpy ndarray
    output_np: np.ndarray = K.tensor_to_image(output)
#Plot the grid of images
    plt.imshow(output_np)  
    plt.axis('off')
    plt.show()     
   
    
    
#%%


#def main(input_filepath=ROOT_PATH+'/data/raw', output_filepath=ROOT_PATH+'/data/processed'):

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


    BATCH_SIZE = 64
    
    ## debug should run below code 
    ROOT_PATH = os.environ.get("ROOT_PATH")
    DATA_PATH = os.environ.get("DATA_PATH_PROC")
    DATA_PATH_TRAIN = os.environ.get("DATA_PATH_PROC")
    os.chdir(ROOT_PATH)
    
    # Load train data
    x = torch.load(DATA_PATH + "/train/images.pt")
    y = torch.load(DATA_PATH + "/train/labels.pt")
    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], y[i]])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    


    img_batch, lab_batch = next(iter(train_loader))
    
    img = img_batch[0]
    
    img2 = K.adjust_brightness(img, 0.1)
    img3 = K.adjust_contrast(img, 0.2)
    img4 = K.adjust_gamma(img, gamma=3., gain=1.5)
    img5 = K.adjust_saturation(img, 0.9)
    img6 = K.adjust_hue(img, 0.5)
#%%
    gaussian = K.filters.GaussianBlur2d((3, 3), (5, 5))
    blur_image: torch.tensor = gaussian(img.unsqueeze(0).float())



#%%
    imshow(blur_image)
#%%    
    imshow(img)
#%%


transform = nn.Sequential(
   K.augmentation.RandomAffine(360),
   K.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3)
   
)

out = transform(img_batch)

#%%

out_img = out[0]
imshow(out_img)
#%%
    # save the training set and corresponding labels to tensors in processed directory 
    images = []
    labels = []
    for image,label in train_loader:
        #print(y)
        
        
        images.append(image)
        labels.append(label)






    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    torch.save(images, output_filepath+'/train/images.pt')
    torch.save(labels, output_filepath+'/train/labels.pt')
    
    # save the trvalidation set and corresponding labels to tensors in processed directory 
    images = []
    labels = []
    for image,label in dataloader_val:
        #print(y)
        images.append(image)
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    torch.save(images, output_filepath+'/val/images.pt')
    torch.save(labels, output_filepath+'/val/labels.pt')
        
    return None # train_set, test_set

#%%
main()