import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN для MNIST/Fashion-MNIST"""

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        features = torch.relu(self.fc1(x))
        logits = self.fc2(features) #последний слой возвращает по одному числу для каждого класса
        if return_features:
            return logits, features
        return logits
