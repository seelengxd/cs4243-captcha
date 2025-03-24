import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.dropout = nn.Dropout(0.3)
        self.fc = spectral_norm(nn.Linear(256 * 4 * 8, 1))
        self.activation = nn.LeakyReLU(0.2, inplace=True) 

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.dropout(x)  
        x = self.activation(self.conv3(x))
        x = self.dropout(x) 
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
