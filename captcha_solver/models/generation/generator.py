import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, text_dim=0):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.embed = nn.Linear(text_dim, 80)
        self.fc = nn.Linear(latent_dim + 80, 512 * 4 * 8)
        self.bn0 = nn.BatchNorm1d(512 * 4 * 8)
        self.conv1 = nn.Conv2d(512, 256 * 4, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1_post = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.bn1_post = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_post = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_post = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1)   
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_post = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_post = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(64)
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z, text=None):
        if text is not None:
            text = self.embed(text)
            x = torch.cat((z, text), dim=1)
        else:
            x = z
        x = F.relu(self.bn0(self.fc(x)), inplace=True) 
        x = x.view(x.size(0), 512, 4, 8)                 
        x = F.relu(self.bn1(F.pixel_shuffle(self.conv1(x), upscale_factor=2)), inplace=True)  
        x = F.relu(self.bn1_post(self.conv1_post(x)), inplace=True)
        x = F.relu(self.bn2(F.pixel_shuffle(self.conv2(x), upscale_factor=2)), inplace=True) 
        x = F.relu(self.bn2_post(self.conv2_post(x)), inplace=True)
        x = F.relu(self.bn3(F.pixel_shuffle(self.conv3(x), upscale_factor=2)), inplace=True)  
        x = F.relu(self.bn3_post(self.conv3_post(x)), inplace=True)
        x = F.relu(self.bn4(F.pixel_shuffle(self.conv4(x), upscale_factor=2)), inplace=True)  
        x = self.res_block1(x)
        x = self.res_block2(x)
        return torch.tanh(self.conv_out(x))
