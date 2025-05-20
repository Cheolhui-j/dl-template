import torch.nn as nn
from src.backbone.resnet import ResNet, BasicBlock

class Classification(nn.Module):
    def __init__(self, num_classes=10, backbone="ResNet"):
        super().__init__()
        if backbone == "ResNet":
            self.feature_extractor = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.feature_extractor(x)