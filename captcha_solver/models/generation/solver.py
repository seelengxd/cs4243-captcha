import torch
import torch.nn as nn
import torch.nn.functional as F

class Solver(nn.Module):
    def __init__(self, num_chars=36, blank_idx=36):
        super(Solver, self).__init__()
        self.num_chars = num_chars
        self.blank_idx = blank_idx 
        self.conv_seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),     
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),      
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),   
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))      
        )
        self.lstm = nn.LSTM(input_size=128*4, hidden_size=256, num_layers=2, 
                             dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_chars + 1)

    def forward(self, x):
        features = self.conv_seq(x)          
        N, C, H, W = features.size()        
        features = features.permute(3, 0, 1, 2).contiguous()
        features = features.view(W, N, C * H)  
        lstm_out, _ = self.lstm(features)      
        T, N, _ = lstm_out.size()
        lstm_out = lstm_out.view(T * N, -1)    
        class_logits = self.fc(lstm_out)      
        class_logits = class_logits.view(T, N, -1)  
        return F.log_softmax(class_logits, dim=2)
