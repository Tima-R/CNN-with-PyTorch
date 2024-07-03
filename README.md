# CNN-with-PyTorch:

A convolutional neural network (CNN) is a kind of neural network that extracts features from matrices of numeric values (often images) by convolving multiple filters over the matrix values to apply weights and identify patterns, such as edges, corners, and so on in an image. The numeric representations of these patterns are then passed to a fully-connected neural network layer to map the features to specific classes.

## Environment:

This project is developed and tested in Google Colab, which provides an excellent platform for running machine learning experiments with access to GPUs and TPUs. Below are the details of the environment and dependencies used:

- **Python Version**: Google Colab provides Python 3.7+
- **PyTorch Version**: 2.3.1
- **Torchvision Version**: 0.18.1
- **Torchaudio Version**: 2.3.1

### Requirements:

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

### Data Exploration:
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

### Load data:
I use the following function to load the dataset, creating training and testing data loaders.

```
# Function to ingest data using training and test loaders
def load_dataset(data_path):
    # Load all of the images
    transformation = transforms.Compose([
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )


    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader


# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset(data_path)
print('Data loaders ready')

```

This function loads all images, applies transformations (converting to tensors and normalizing), splits the data into training (70%) and testing (30%) sets, and creates iterative data loaders for batch processing.


### Define the CNN:
A convolutional neural network (CNN) model using PyTorch is defined. The architecture includes three convolutional layers with ReLU activations and max pooling, followed by a dropout layer to prevent overfitting. The final fully connected layer maps the extracted features to class probabilities.

```
# Create a neural net class
class Net(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        
        # Our images are RGB, so input channels = 3. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # So our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to  the probability for each class
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
      
        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return log_softmax tensor 
        return F.log_softmax(x, dim=1)
    
print("CNN model class defined!")

```

### Training and Testing the Model
A defined functions to train and test the convolutional neural network model using the training and testing datasets. The training function processes the images in batches, calculates the loss, and updates the model weights. The testing function evaluates the model's performance on the validation set and calculates the accuracy.

```
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)
        
        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics for every 10 batches so we see some progress
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
            
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
    
    
# Now use the train and test functions to train and test the model    

device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)

# Create an instance of the model class and allocate it to the device
model = Net(num_classes=len(classes)).to(device)

# Use an "Adam" optimizer to adjust weights
# (see https://pytorch.org/docs/stable/optim.html#algorithms for details of supported algorithms)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 5 epochs (in a real scenario, you'd likely use many more)
epochs = 5
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

```

### View the loss history
To track average training and validation loss for each epoch. To plot these to verify that loss reduced as the model was trained, and to detect over-fitting (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase).

![image](https://github.com/Tima-R/CNN-with-PyTorch/assets/116596345/7e7c1210-7352-4565-996f-97e2034d6dc7)

The image shows a line plot of loss values over epochs for both training and validation data. The x-axis represents the number of epochs (ranging from 1 to 5), and the y-axis represents the loss. The plot has two lines: one for training loss (in blue) and one for validation loss (in orange). Both lines show a decreasing trend, indicating that the loss is reducing over the epochs, with the validation loss consistently lower than the training loss throughout the epochs. This suggests that the model is learning and improving its performance over time.


### Model Evaluation
To evaluate the model's performance, the confusion matrix is used to visualize the accuracy of predictions on the test set. The following code demonstrates how to generate and plot the confusion matrix using SciKit-Learn:

```
# Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
from sklearn.metrics import confusion_matrix

# Set the model to evaluate mode
model.eval()

# Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in test_loader:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()

```
![image](https://github.com/Tima-R/CNN-with-PyTorch/assets/116596345/bac59839-6b19-42f9-95c8-5c2385ca84fc)


The confusion matrix shows the performance of a shape classification model with actual shapes (circle, square, triangle) on the y-axis and predicted shapes on the x-axis. Darker cells indicate higher counts. The model predicts "circle" and "triangle" shapes accurately, with high counts on the diagonal for these categories. However, there is moderate misclassification for "squares," indicating some prediction errors.


### Saving the Trained Model
After training the model, to save the model weights to a file for later use. The following code demonstrates how to save the trained model weights using PyTorch:

```
# Save the model weights
model_file = 'models/shape_classifier.pt'
torch.save(model.state_dict(), model_file)
del model
print('model saved as', model_file)

```
























