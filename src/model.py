import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50()
        
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    
    return model