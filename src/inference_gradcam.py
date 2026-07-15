import argparse
from pathlib import Path

from PIL import Image

from inference import (
    DEFAULT_CHECKPOINT_PATH,
    get_device,
    gradcam_heatmap,
    load_model,
    overlay_heatmap,
    predict,
    preprocess_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Инференс ResNet-50 и визуализация Grad-CAM для рентгеновского снимка."
    )
    parser.add_argument("--image", required=True, type=Path, help="Путь к PNG/JPG/JPEG снимку.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint, созданный train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "gradcam_result.png",
        help="Куда сохранить изображение с тепловой картой.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Устройство для инференса."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {args.image}")

    device = get_device(args.device)
    model, class_names = load_model(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    image_tensor = preprocess_image(image)

    predicted_index, probabilities = predict(model, image_tensor, device)
    heatmap = gradcam_heatmap(model, image_tensor, predicted_index, device)
    result = overlay_heatmap(image, heatmap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)

    print(f"Предсказание: {class_names[predicted_index]} ({probabilities[predicted_index]:.1%})")
    print("Вероятности: " + ", ".join(f"{name}={value:.1%}" for name, value in zip(class_names, probabilities)))
    print(f"Grad-CAM сохранён: {args.output.resolve()}")


if __name__ == "__main__":
    main()
