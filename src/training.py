"""Общие функции обучения и получения предсказаний"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .models import SimpleCNN
from .utils import set_seed, get_device


def train_one_epoch(model, loader, optimizer):
    device = next(model.parameters()).device
    model.train()
    loss_function = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()


def predict(model, X, batch_size=128):
    model.eval() #модель в режим оценки
    device = next(model.parameters()).device
    all_probs = []
    all_features = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            images = X[start:start + batch_size].to(device, non_blocking=(device.type == "cuda"))
            logits, features = model(images, return_features=True)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy()) #логиты в вероятности
            all_features.append(features.cpu().numpy())
    return np.vstack(all_probs), np.vstack(all_features)


def measure_cnn_quality(
    X_fit, y_fit, X_eval, y_eval, epochs=100, seed=42,
    num_classes=10, batch_size=128, device=None
):
    """Обучает CNN и считает метрики на тестовой выборке"""
    set_seed(seed)
    device = get_device() if device is None else torch.device(device)
    model = SimpleCNN(in_channels=X_fit.shape[1], num_classes=num_classes).to(device)
    loader = DataLoader(
        TensorDataset(X_fit, y_fit), batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda")
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loader, optimizer)
        if epoch % 25 == 0 or epoch == epochs:
            print(f"Эпоха {epoch}/{epochs}")

    probs, _ = predict(model, X_eval, batch_size=batch_size)
    predictions = probs.argmax(axis=1)
    targets = y_eval.cpu().numpy()

    return {
        "Accuracy": accuracy_score(targets, predictions),
        "Balanced accuracy": balanced_accuracy_score(targets, predictions),
        "Macro-F1": f1_score(targets, predictions, average="macro")
    }


def train_and_measure(
    X_train_local, y_train_local, X_test_local, y_test_local,
    num_classes=10, epochs=3, seed=42, batch_size=128, device=None
):
    """Обучает новый классификатор и возвращает точность на чистом test"""
    set_seed(seed)
    device = get_device() if device is None else torch.device(device)

    model = SimpleCNN(in_channels=X_train_local.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        TensorDataset(X_train_local, y_train_local),
        batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda")
    )

    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer)

    probs, _ = predict(model, X_test_local, batch_size=batch_size)
    accuracy = (probs.argmax(axis=1) == y_test_local.cpu().numpy()).mean()
    return float(accuracy)
