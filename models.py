## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #LeNet-5 Architecture
        
        # Input image size: 224x224
        # 1 input image channel (grayscale), 6 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (6, 220, 220)
        # after one pool layer, this becomes (6, 110, 110)
        self.conv1 = nn.Conv2d(1, 6, 5)
        
        # 16 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (16, 106, 106)
        # after one pool layer, this becomes (16, 53, 53)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # 120 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (120, 49, 49)
        self.conv3 = nn.Conv2d(16, 120, 5)
        
        # Average Pooling Layer with Kernel and Stride equals to 2
        self.pool = nn.AvgPool2d(2, 2)
        
        self.fc1 = nn.Linear(120*49*49,512)
        
        self.fc2 = nn.Linear(512,136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.tanh(self.conv1(x)))
        
        x = self.pool(F.tanh(self.conv2(x)))
        
        x = F.tanh(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.tanh(self.fc1(x))
        
        # Output without Softmax
        x = self.fc2(x)
        
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #Custom LeNet-5 Architecture
        
        # Input image size: 224x224
        # 1 input image channel (grayscale), 32 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # 64 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (64, 106, 106)
        # after one pool layer, this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # 128 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (128, 49, 49)
        # after one pool layer, this becomes (128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # 256 output channels/feature maps and 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (24-5)/1 +1 = 20
        # the output Tensor for one image, will have the dimensions: (256, 20, 20)
        # after one pool layer, this becomes (256, 10, 10)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        
        # Average Pooling Layer with Kernel and Stride equals to 2
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*10*10,512)
        
        self.fc2 = nn.Linear(512,136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        # Output without Softmax
        x = self.fc2(x)
        
        return x