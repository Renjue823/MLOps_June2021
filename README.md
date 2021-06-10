# MLOps_June2021

### Our Team
- Freja Byskov Kristensen (S174464)
- [Anna Schibelle](https://github.com/schibsen) (S166154)
- [Laura Bonde Holst](https://github.com/s173953) (S173953)
- [Renjue Sun](https://github.com/Renjue823) (S181294)

### Dataset 
We will be using the Animal Faces dataset from Yunjey Choi, Youngjung Uh, Jaejun Yoo and Jung-Woo Ha. (https://www.kaggle.com/andrewmvd/animal-faces) 

### The model
Deep learning model to classify whether the animal is a dog, cat or wildlife. We intent to use dropout, pooling, different optimisers and loss functions. 

### Chosen framework
We will be working with Kornia, which will be used within the model to extract features and detect edges. Potentially we will use the loss function from Kornia. As the dataset has over 15,000 datapoints, we will not prioritise using Kornia for data augmentation. 

## Environment
### Creating a new environment using environment.yml
When running first time you can creat an environment form the .yml file by running 

`$ conda env create -f environment.yml`


### Updating the environment according to environment.yml 
I have not tried this piece of code yet, but if the environment.yml doesn't hold any packages, exclude --prune

`$ conda env update --prefix ./env --file environment.yml  --prune`

### Updating environment.yml 
When you install new packages crucial to the workflow, remember to update the environment.yml

`$ conda env export > environment.yml`


