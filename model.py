import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class ProjectionNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(ProjectionNet, self).__init__()
        #self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
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
        # last_layer = 512
        # sequential_layers = []
        # for num_neurons in head_layers:
        #     sequential_layers.append(nn.Linear(last_layer, num_neurons))
        #     sequential_layers.append(nn.BatchNorm1d(num_neurons))
        #     sequential_layers.append(nn.ReLU(inplace=True))
        #     last_layer = num_neurons
        # head = nn.Sequential(
        #     sequential_layers)
        # self.head = head
        # self.out = nn.Linear(last_layer, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def freeze_vgg(self):
        # freez full vgg16.features
        for param in self.features.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        # for param in self.classifier.parameters():
        #     param.requires_grad = True
            
    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
