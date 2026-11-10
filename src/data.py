#ДАННЫЕ И МОДЕЛЬ

import torch
from torchvision import datasets


def load_small_dataset(dataset_name, train_limit, test_limit, data_root, seed=42):
    """Загружает MNIST или Fashion-MNIST и выбирает небольшие подмножества."""
    if dataset_name.lower() == "mnist":
        DatasetClass = datasets.MNIST
    elif dataset_name.lower() == "fashionmnist":
        DatasetClass = datasets.FashionMNIST
    else:
        raise ValueError("DATASET_NAME должен быть MNIST или FashionMNIST")

    train_data = DatasetClass(root=str(data_root), train=True, download=True)
    test_data = DatasetClass(root=str(data_root), train=False, download=True)

    generator = torch.Generator().manual_seed(seed)

    #случайная перестановка чисел от 0 до len(train_data.targets)-1, берем первые ..._limit
    train_ids_all = torch.randperm(len(train_data.targets), generator=generator)
    train_ids = train_ids_all if train_limit is None else train_ids_all[:train_limit]
    test_ids = (torch.arange(len(test_data.targets)) if test_limit is None
                else torch.randperm(len(test_data.targets), generator=generator)[:test_limit])

    #выбирается изображение, добавляется 1 измерение(1 канал) и нормализуются пиксели
    X_train = train_data.data[train_ids].float().unsqueeze(1) / 255.0
    X_test = test_data.data[test_ids].float().unsqueeze(1) / 255.0

    #преобразование меток в лонг и выбор определенных меток
    y_train = torch.as_tensor(train_data.targets, dtype=torch.long)[train_ids]
    y_test = torch.as_tensor(test_data.targets, dtype=torch.long)[test_ids]

    return X_train, y_train, X_test, y_test
