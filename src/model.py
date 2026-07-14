import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50()
        
    # Меняем финальный слой. У ResNet-50 на входе в fc слой 2048 признаков
    in_features = model.fc.in_features
    
    # Кастомная голова: Dropout помогает против переобучения
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2) # 2 класса: норма и пневмония
    )
    
    return model