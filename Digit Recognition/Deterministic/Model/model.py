import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()       
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())      
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)       
        return x

class mlp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(mlp, self).__init__()        
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_ch))

    def forward(self, x):
        x = self.fc(x)        
        return x


class FCNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=10):
        super(FCNet, self).__init__()
        filters = [32, 64]
        
        self.conv1_1 = nn.Conv2d(in_ch, filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(filters[0])
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(filters[1])
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)   
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters[1]*7*7, 1024, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, out_ch, bias=True)
        
    def forward(self, x, mask=True):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)        
        x = self.fc2(x)
        return x