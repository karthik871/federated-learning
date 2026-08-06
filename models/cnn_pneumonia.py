import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class PneumoniaModel(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)