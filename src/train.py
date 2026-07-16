from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from dataset import ChestXRayDataset, get_transforms
from model import build_model


CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def calculate_class_weights(labels, device):
    """Return inverse-frequency weights aligned with the dataset class indices."""
    class_counts = torch.bincount(
        torch.as_tensor(labels, dtype=torch.long), minlength=len(CLASS_NAMES)
    ).float()
    if torch.any(class_counts == 0):
        missing = [CLASS_NAMES[i] for i, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"В обучающем наборе отсутствуют классы: {', '.join(missing)}")

    # N / (K * n_c): the average weight is one, while rare classes get more weight.
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts)
    return class_weights.to(device), class_counts

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, f1

def main():
    print("Torch version:", torch.version)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"

    print(f"Running on {DEVICE} with {torch.get_num_threads()} threads")

    train_transform, val_transform = get_transforms()
    train_dataset = ChestXRayDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = ChestXRayDataset(VAL_DIR, transform=val_transform)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Не удалось найти изображения в data/train или data/val.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_weights, class_counts = calculate_class_weights(train_dataset.labels, DEVICE)
    print(
        "Class counts: "
        + ", ".join(
            f"{name}={int(count)}" for name, count in zip(CLASS_NAMES, class_counts)
        )
    )
    print(
        "Class weights: "
        + ", ".join(
            f"{name}={weight:.4f}" for name, weight in zip(CLASS_NAMES, class_weights)
        )
    )

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "models" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    
    mlflow.set_experiment("Pneumonia_Detection_ResNet50")
    
    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "CrossEntropyLoss(weight=class_weights)")
        mlflow.log_dict(
            {
                "class_names": list(CLASS_NAMES),
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights.detach().cpu().tolist(),
            },
            "class_distribution.json",
        )
        
        best_f1 = float("-inf")
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_f1 = validate(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": list(CLASS_NAMES),
                        "class_weights": class_weights.detach().cpu(),
                        "val_f1": best_f1,
                    },
                    checkpoint_path,
                )
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="best_model_resnet50",
                    registered_model_name="pneumonia-resnet50",
                    serialization_format="pickle"
                )
                
        print(f"Обучение завершено. Лучший Val F1-score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
