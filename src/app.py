from pathlib import Path
from PIL import Image
import streamlit as st

from inference import (
    DEFAULT_CHECKPOINT_PATH,
    get_device,
    gradcam_heatmap,
    load_model,
    overlay_heatmap,
    predict,
    preprocess_image,
)

RUSSIAN_LABELS = {"NORMAL": "Норма", "PNEUMONIA": "Пневмония"}

st.set_page_config(page_title="Анализ рентгена лёгких", page_icon="🫁", layout="wide")
st.title("🫁 Определение пневмонии по рентгеновскому снимку")
st.caption(
    "Модель ResNet‑50 анализирует рентген грудной клетки и показывает области, "
    "повлиявшие на предсказание, с помощью Grad-CAM."
)
st.warning(
    "Это демонстрационный инструмент, а не медицинское заключение. "
    "Результат должен интерпретировать врач."
)


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path: str):
    device = get_device()
    model, class_names = load_model(checkpoint_path, device)
    return model, class_names, device


with st.sidebar:
    st.header("Модель")
    checkpoint_path = st.text_input("Путь к checkpoint", str(DEFAULT_CHECKPOINT_PATH))
    st.caption("Checkpoint создаётся при запуске `python src/train.py`.")

uploaded_file = st.file_uploader(
    "Загрузите рентгеновский снимок", type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Поддерживаются файлы PNG, JPG и JPEG.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except OSError:
    st.error("Не удалось прочитать загруженный файл как изображение.")
    st.stop()

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Исходный снимок")
    st.image(image, use_container_width=True)

try:
    with st.spinner("Выполняется анализ снимка..."):
        model, class_names, device = get_model(str(Path(checkpoint_path).expanduser()))
        image_tensor = preprocess_image(image)
        predicted_index, probabilities = predict(model, image_tensor, device)
        heatmap = gradcam_heatmap(model, image_tensor, predicted_index, device)
        explanation = overlay_heatmap(image, heatmap)
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(str(error))
    st.stop()

predicted_class = class_names[predicted_index]
label = RUSSIAN_LABELS.get(predicted_class, predicted_class)
confidence = float(probabilities[predicted_index])

with right_column:
    st.subheader("Grad-CAM")
    st.image(explanation, use_container_width=True)
    st.caption("Красно-жёлтые области сильнее повлияли на предсказание модели.")

st.subheader(f"Результат: {label}")
st.metric("Уверенность модели", f"{confidence:.1%}")

for index, class_name in enumerate(class_names):
    st.progress(float(probabilities[index]), text=f"{RUSSIAN_LABELS.get(class_name, class_name)}: {probabilities[index]:.1%}")
