import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.fc = nn.Sequential(
            nn.Linear(25088, 1024), 
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

    def get_embedding(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, input_1, input_2):
        out_1 = self.get_embedding(input_1)
        out_2 = self.get_embedding(input_2)
        return out_1, out_2