import torch.nn as nn
from src.backbone.resnet import build_resnet

class Classification(nn.Module):
    def __init__(self, num_classes=10, backbone="resnet18"):
        super().__init__()
        self.backbone=build_resnet(name=backbone, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)