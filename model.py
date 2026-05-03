import torch
import torch.nn as nn
import torchvision.models as models


class MultiLabelModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_small(pretrained=True)


        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x