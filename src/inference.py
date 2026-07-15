from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch

from dataset import get_transforms
from model import build_model


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pt"


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model(checkpoint_path: str | Path, device: torch.device):
    """Load a checkpoint created by train.py and return an evaluation-ready model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint не найден: {checkpoint_path}. "
            "Сначала запустите обучение: python src/train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    class_names: Sequence[str] = checkpoint.get("class_names", CLASS_NAMES)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tuple(class_names)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize and normalize a chest X-ray exactly as in validation/inference."""
    _, val_transform = get_transforms()
    image_array = np.asarray(image.convert("RGB"))
    return val_transform(image=image_array)["image"].unsqueeze(0)


def predict(model, image_tensor: torch.Tensor, device: torch.device):
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    return predicted_index, probabilities


def gradcam_heatmap(
    model, image_tensor: torch.Tensor, target_class: int, device: torch.device
) -> np.ndarray:
    """Compute Grad-CAM for the final ResNet convolutional layer."""
    activations = []
    gradients = []
    target_layer = model.layer4[-1].conv3

    forward_handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output.detach())
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0].detach())
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor.to(device))
        logits[0, target_class].backward()

        if not activations or not gradients:
            raise RuntimeError("Не удалось получить активации или градиенты для Grad-CAM.")

        channel_weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        heatmap = (channel_weights * activations[-1]).sum(dim=1).relu()[0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        return heatmap.cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a JET heatmap on the original RGB image without changing its size."""
    image_array = np.asarray(image.convert("RGB"))
    height, width = image_array.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    colored_heatmap = cv2.applyColorMap(
        np.uint8(np.clip(resized_heatmap, 0, 1) * 255), cv2.COLORMAP_JET
    )
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_array, 1 - alpha, colored_heatmap, alpha, 0)
    return Image.fromarray(overlay)
