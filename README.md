# Pneumonia Classifier
Веб-приложение для демонстрационного определения пневмонии на рентгеновских снимках грудной клетки. В основе — дообученная `ResNet-50`; для борьбы с дисбалансом классов используется взвешенная кросс-энтропия, а Grad-CAM показывает области изображения, сильнее всего повлиявшие на решение модели.
## Приложение в Streamlit Community Cloud
https://pneumoniacl.streamlit.app/
> **Важно:** это учебно-исследовательский проект, а не медицинское изделие и не средство постановки диагноза. Предсказание модели нельзя использовать вместо заключения врача.
## Возможности

- Обучение `ResNet-50` для двух классов: `NORMAL` и `PNEUMONIA`.
- Аугментации и нормализация входных снимков с помощью Albumentations.
- Компенсация дисбаланса классов через `nn.CrossEntropyLoss(weight=class_weights)`.
- Сохранение лучшего checkpoint в `models/best_model.pt` по macro F1-score на валидации.
- Логирование гиперпараметров, метрик и checkpoint в MLflow.
- Streamlit-интерфейс для загрузки PNG/JPG/JPEG, вывода вероятностей и Grad-CAM.
- Отдельный скрипт, сохраняющий Grad-CAM-наложение в PNG.

## Структура проекта

```text
pneumonia_classifier/
├── data/                         # Датасет (не хранится в Git)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
│   └── best_model.pt             # Лучший checkpoint для инференса
├── outputs/                      # Результаты Grad-CAM (игнорируется Git)
├── src/
│   ├── app.py                    # Streamlit-приложение
│   ├── dataset.py                # Dataset и аугментации
│   ├── inference.py              # Общие функции инференса и Grad-CAM
│   ├── inference_gradcam.py      # CLI-инференс с сохранением тепловой карты
│   ├── model.py                  # ResNet-50 с кастомной классификационной головой
│   └── train.py                  # Обучение, оценка и MLflow-логирование
├── requirements.txt
└── README.md
```

## Требования

- Python 3.10 или новее.
- Для ускорения обучения рекомендуется NVIDIA GPU с установленным CUDA-совместимым PyTorch, но инференс и обучение работают и на CPU.
- Рентгеновские изображения в форматах `.png`, `.jpg` или `.jpeg`.

## Установка

Клонируйте репозиторий и перейдите в его папку:

```bash
git clone https://github.com/BeachlandBallroom/pneumonia_classifier.git
cd pneumonia_classifier
```

Создайте и активируйте виртуальное окружение.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установите зависимости:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Данные

Скрипт обучения ожидает следующую структуру директорий:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Имена папок важны: они задают соответствие индексов классов — `0 = NORMAL`, `1 = PNEUMONIA`.

Встроенные преобразования:

- обучение: resize до `224×224`, горизонтальное отражение, изменение яркости/контраста, сдвиг/масштаб/поворот и нормализация ImageNet;
- валидация и инференс: resize до `224×224` и та же нормализация ImageNet без случайных аугментаций.

## Обучение

Запустите команду из корневой папки репозитория:

```bash
python src/train.py
```

По умолчанию используются:

| Параметр | Значение |
| --- | --- |
| Архитектура | ResNet-50 с предобученными ImageNet-весами |
| Размер изображения | 224 × 224 |
| Batch size | 64 |
| Количество эпох | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Метрика выбора checkpoint | Macro F1 на validation-наборе |

### Борьба с дисбалансом классов

Веса рассчитываются только по меткам обучающей выборки. Для каждого класса `c` используется формула:

```text
weight[c] = N / (K × count[c])
```

где `N` — количество всех обучающих примеров, `K` — число классов, `count[c]` — число примеров класса. Таким образом, редкий класс получает больший вклад в значение функции потерь.

В коде это выглядит так:

```python
class_weights, class_counts = calculate_class_weights(train_dataset.labels, DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Скрипт выводит размеры классов и рассчитанные веса в консоль, а также сохраняет их в MLflow.

### Результаты обучения

При каждом улучшении macro F1 на validation-наборе `train.py`:

1. сохраняет state dict и метаданные в `models/best_model.pt`;
2. добавляет checkpoint в артефакты MLflow: `checkpoints/best_model.pt`;
3. логирует полную PyTorch-модель в MLflow под именем `pneumonia-resnet50`.

Локальный MLflow tracking URI указывает на SQLite-базу `mlflow.db` в корне репозитория. Чтобы открыть интерфейс MLflow после обучения, выполните:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

После этого откройте `http://127.0.0.1:5000` в браузере.

## Streamlit-интерфейс

Перед первым запуском интерфейса убедитесь, что существует `models/best_model.pt`. Он уже добавлен в репозиторий; при переобучении автоматически перезаписывается лучшей моделью.

Запустите приложение из корня проекта:

```bash
streamlit run src/app.py
```

Обычно Streamlit откроет страницу `http://localhost:8501`.

В интерфейсе:

1. При необходимости укажите путь к checkpoint в левой боковой панели.
2. Загрузите рентгеновский снимок в формате PNG, JPG или JPEG.
3. Получите предсказанный класс, уверенность модели и вероятности для обоих классов.
4. Посмотрите Grad-CAM: красно-жёлтые области — участки, которые сильнее повлияли на результат модели.

## Grad-CAM из командной строки

Для инференса одного снимка и сохранения тепловой карты используйте:

```bash
python src/inference_gradcam.py \
  --image path/to/xray.jpeg \
  --checkpoint models/best_model.pt \
  --output outputs/gradcam_result.png
```

В Windows PowerShell используйте обратный апостроф для переноса строки или напишите команду в одну строку:

```powershell
python src/inference_gradcam.py --image "data\test\NORMAL\IM-0001-0001.jpeg" --checkpoint "models\best_model.pt" --output "outputs\gradcam_result.png"
```

Дополнительные параметры:

```text
--device auto   # Автоматически выбрать CUDA при её наличии, иначе CPU
--device cpu    # Принудительно использовать CPU
--device cuda   # Принудительно использовать GPU
```

Пример:

```bash
python src/inference_gradcam.py --image xray.jpg --device cpu
```

По умолчанию результат записывается в `outputs/gradcam_result.png`.

## Как работает Grad-CAM

Grad-CAM строится для слоя `model.layer4[-1].conv3` — последнего свёрточного слоя ResNet-50:

1. выполняется прямой проход для изображения;
2. выбирается логит предсказанного класса;
3. вычисляется градиент логита по активациям целевого слоя;
4. средние значения градиентов по каналам используются как веса активаций;
5. полученная карта нормализуется, масштабируется до исходного размера снимка и накладывается палитрой JET.

Grad-CAM показывает, какие области повлияли на решение нейросети, но не доказывает наличие или отсутствие патологий.

## Деплой в Streamlit Community Cloud

Репозиторий подготовлен к деплою:

- входной файл: `src/app.py`;
- зависимости: `requirements.txt` в корне;
- checkpoint: `models/best_model.pt` хранится в репозитории.

В [Streamlit Community Cloud](https://share.streamlit.io/) создайте приложение со следующими параметрами:

| Поле | Значение |
| --- | --- |
| Repository | `BeachlandBallroom/pneumonia_classifier` |
| Branch | `main` |
| Main file path | `src/app.py` |

После создания приложения каждый `git push origin main` автоматически запустит обновление облачной версии. Если меняются библиотеки, не забудьте закоммитить и `requirements.txt`.

Для обновления checkpoint:

```bash
git add models/best_model.pt
git commit -m "Update trained model"
git push origin main
```

Checkpoint имеет большой размер. При частых обновлениях модели рекомендуется использовать Git LFS.

## Частые проблемы

### `Checkpoint не найден`

Запустите обучение:

```bash
python src/train.py
```

или убедитесь, что в поле боковой панели Streamlit указан путь к существующему `.pt`-файлу.

### `ModuleNotFoundError`

Активируйте виртуальное окружение и повторно установите зависимости:

```bash
python -m pip install -r requirements.txt
```

### PyTorch не использует GPU

Проверьте в Python:

```python
import torch
print(torch.cuda.is_available())
```

Если будет `False`, установите сборку PyTorch, соответствующую вашей версии CUDA, или используйте CPU.

### Streamlit Cloud не находит модель

Проверьте, что `models/best_model.pt` добавлен в Git и отправлен в ветку, выбранную при деплое:

```bash
git ls-files models/best_model.pt
```

## Лицензия и использование

Перед публичным использованием проверьте лицензию исходного датасета, права на медицинские изображения и применимые требования к обработке медицинских данных. Не загружайте в публичное приложение снимки с персональными данными пациента.
