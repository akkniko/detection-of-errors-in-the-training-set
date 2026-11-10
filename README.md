# Проект ВКР: поиск ошибочных меток

## Структура

```text
ВКР1_8_проект/
├── ВКР1_8_сокращенный_эксперимент.ipynb
├── requirements.txt
├── data/                 # сюда скачиваются датасеты
├── results_vkr/          # сюда сохраняются CSV и графики
└── src/
    ├── __init__.py
    ├── data.py           # загрузка MNIST и Fashion-MNIST
    ├── training.py       # обучение CNN, предсказания и оценка качества
    ├── utils.py          # seed и выбор устройства
    └── models/
        ├── __init__.py
        ├── cnn.py        # исходная архитектура SimpleCNN
        └── fashion_cnn.py # альтернативная FashionCNN
```

## Как открыть проект в VS Code

1. Распакуйте архив в отдельную папку
2. В VS Code выберите **File → Open Folder** и откройте именно папку `ВКР1_8_проект`
3. Откройте ноутбук `ВКР1_8_сокращенный_эксперимент.ipynb`
4. Выберите Python-интерпретатор, в котором установлены зависимости
5. Запускайте ячейки сверху вниз

Ноутбук ищет папку `src` относительно текущей рабочей папки и добавляет корень проекта в `sys.path`. Поэтому важно открывать в VS Code корневую папку проекта, а не только сам файл ноутбука.

## Установка зависимостей

Сначала установите CUDA-сборку `torch` и `torchvision` через официальный селектор PyTorch: https://pytorch.org/get-started/locally/

Не устанавливайте `torch` и `torchvision` из обычного списка зависимостей поверх уже настроенной CUDA-сборки, чтобы случайно её не заменить

После этого в терминале VS Code выполните:

```powershell
python -m pip install -r requirements.txt
```

Если CUDA не доступна, ноутбук остановится в начале, потому что в настройках `REQUIRE_CUDA = True`.

## Что вынесено из ноутбука

- `SimpleCNN` — архитектура модели
- `load_small_dataset` — загрузка и подготовка MNIST/Fashion-MNIST
- `train_one_epoch` — один проход обучения
- `predict` — получение вероятностей классов и признаков
- `measure_cnn_quality` — обучение базовой CNN и расчёт Accuracy, Balanced Accuracy и Macro-F1
- `train_and_measure` — обучение классификатора и расчёт Accuracy после очистки
- `set_seed` и `get_device` — воспроизводимость и выбор CPU/GPU

В ноутбуке оставлены этапы исследования: генерация шума, OOF, Cleanlab, Combined, SEI, подбор tau, абляции и сравнение вариантов очистки. Их логика ещё может меняться в ходе ВКР, поэтому пока она остаётся рядом с экспериментами.

## Как импортировать модули вручную

В первой импортной ячейке ноутбука уже стоят такие импорты:

```python
from src.models import SimpleCNN
from src.data import load_small_dataset
from src.training import train_one_epoch, predict, measure_cnn_quality, train_and_measure
from src.utils import set_seed, get_device
```

Например, чтобы создать модель:

```python
model = SimpleCNN(in_channels=1, num_classes=10).to(DEVICE)
```

Чтобы загрузить датасет:

```python
X_train, y_true, X_test, y_test = load_small_dataset(
    DATASET_NAME, TRAIN_LIMIT, TEST_LIMIT, data_root=DATA_ROOT, seed=SEED
)
```


## Сравнение с альтернативной FashionCNN

В конец ноутбука добавлен этап 10. Он:

- проверяет базовое качество FashionCNN на 3 000 и 10 000 изображениях;
- сравнивает FashionCNN и SimpleCNN по Accuracy, Balanced Accuracy и Macro-F1;
- повторно обучает FashionCNN на тех же вариантах данных после очистки, которые уже сформированы этапом 8.4;
- сравнивает влияние очистки на обе архитектуры.

Для этого этапа сначала необходимо выполнить ноутбук сверху вниз до конца этапа 9. Этап 10 использует `replicate_cache` с результатами этапа 8.4 и не запускает OOF/SEI повторно. Входы FashionCNN нормализуются по mean/std текущей обучающей выборки. Размеры наборов, тестовая выборка, seed и число эпох остаются такими же, как в основном эксперименте.

В папке `results_vkr` появятся файлы `fashion_cnn_quality_runs.csv`, `architecture_quality_comparison.csv`, `fashion_cnn_cleaning_runs.csv` и `cleaning_architecture_comparison.csv` со сводными результатами.
