import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification

    def forward(self, x):
        return self.model(x)
