import torch.nn as nn


# Simple CNN model

class CNNVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        self.model.append(nn.Conv2d(3,32,[3,3]))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d([2,2],[2,2]))
        self.model.append(nn.Conv2d(32,32,[5,5]))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d([2,2],[2,2]))
        self.model.append(nn.Conv2d(32,32,[7,7]))
        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(32,8,[1,1]))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d([2,2],[2,2]))
        self.model.append(nn.Flatten())

    def add_linear_layer(self,size,n_classes = 10):
        self.model.append(nn.Linear(size,n_classes))

    def forward(self,x):
        out = self.model(x)
        return out

