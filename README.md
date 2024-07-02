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

```
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
```

## Below are the libraries used:

```
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
```

### Uploading Local Folder to Google Colab:
To upload a local folder to Google Colab, follow these steps:

Download the zip file from Kaggle.
Zip the folder on your local PC.
Upload the zip file to Colab using files.upload().
Unzip the file in Colab using zipfile.ZipFile

```
import zipfile
import os
```

### Data Exploration
To understand the dataset better, I display the first image from each category. Below is an example image showing the first image from three different shape categories: Square, Triangle, and Circle.

```
# Show the first image in each folder
fig = plt.figure(figsize=(8, 12))
i = 0
for sub_dir in os.listdir(data_path):
    i += 1
    img_file = os.listdir(os.path.join(data_path, sub_dir))[0]
    img_path = os.path.join(data_path, sub_dir, img_file)
    img = mpimg.imread(img_path)
    a = fig.add_subplot(1, len(classes), i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file)
plt.show()

```
This code iterates through each subdirectory in the dataset, reads the first image file, and displays it in a matplotlib figure, providing a quick visual overview of the dataset.

![image](https://github.com/Tima-R/CNN-with-PyTorch/assets/116596345/d31f8d82-562b-4682-8233-1b773aeac102)


The images are labeled with their respective shapes and an identifier. This visualization helps us verify the contents and structure of the dataset, ensuring that the images are correctly categorized and loaded. Each subplot title shows the shape and its corresponding filename. This is useful for a quick visual inspection of the dataset.














































