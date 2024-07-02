# CNN-with-PyTorch

A convolutional neural network (CNN) is a kind of neural network that extracts features from matrices of numeric values (often images) by convolving multiple filters over the matrix values to apply weights and identify patterns, such as edges, corners, and so on in an image. The numeric representations of these patterns are then passed to a fully-connected neural network layer to map the features to specific classes.

## Environment

This project is developed and tested in Google Colab, which provides an excellent platform for running machine learning experiments with access to GPUs and TPUs. Below are the details of the environment and dependencies used:

- **Python Version**: Google Colab provides Python 3.7+
- **PyTorch Version**: 2.3.1
- **Torchvision Version**: 0.18.1
- **Torchaudio Version**: 2.3.1

### Requirements

To ensure compatibility and reproducibility, the following versions of PyTorch and its related libraries are used:

```sh
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1

## Below are the libraries used:

```sh
# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# Other libraries we'll use
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

### Uploading Local Folder to Google Colab:
To upload a local folder to Google Colab, follow these steps:

1- Download the zip file from Kaggle.
2- Zip the folder on your local PC.
3- Upload the zip file to Colab using files.upload().
4- Unzip the file in Colab using zipfile.ZipFile

