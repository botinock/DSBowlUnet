# DSBowlUnet
https://www.kaggle.com/c/data-science-bowl-2018 

U-Net test task for Quantum
______________________________________
## Description
The project was created as a test task for intership in Quantum. The task is semantic segmentation of the nuclei.
### Model
The provided model named U-Net is basically a convolutional auto-encoder with skip connections from encoder
layers to decoder layers that are on the same level.

![U-net](https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018/blob/master/u-net-architecture.png)

Model was created using Python framework Keras.

## How to run the code
First of all you should extract archives with images. To do this, just run **extract.py**. The script creates folders and
extracts archives there. Next, for training of the model you need to run **data_utils.py** then **train.py** or instantly **train.py**. 
**data_utils.py** saves the data in convenient **.npy** format. **train.py** loads the data and starts learning. At the end of learning,
script will save the model. To predict, you should run **predict_mask.py**. The script will save masks in **.npy** format. 
