import torch
import torch.nn as nn
from torchvision.models import vgg16


class Vgg(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(Vgg, self).__init__()
        self.features = vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
