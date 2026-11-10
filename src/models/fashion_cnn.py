import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """Компактная CNN для классификации Fashion-MNIST."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.features(x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        features = self.classifier[2](x)
        x = self.classifier[3](features)
        logits = self.classifier[4](x)

        if return_features:
            return logits, features
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
