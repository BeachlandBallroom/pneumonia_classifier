import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np

from dataset import ChestXRayDataset, get_transforms
from model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Оборачиваем dataloader в tqdm
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
        
        # Обновляем информацию в консоли на лету
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
    # --- Настройки Гиперпараметров ---
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Пути к данным (замени на свои)
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    
    # --- Подготовка данных и модели ---
    train_transform, val_transform = get_transforms()
    train_dataset = ChestXRayDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = ChestXRayDataset(VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Locate the database relative to this repository, not to a specific computer.
    project_root = Path(__file__).resolve().parents[1]
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    
    # --- Интеграция с MLflow ---
    mlflow.set_experiment("Pneumonia_Detection_ResNet50")
    
    with mlflow.start_run():
        # Логируем параметры обучения
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("optimizer", "Adam")
        
        best_f1 = 0.0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_f1 = validate(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Логируем метрики за каждую эпоху
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            
            # Сохраняем лучшую модель по метрике F1-score
            if val_f1 > best_f1:
                best_f1 = val_f1
                # Логируем саму модель PyTorch прямо в MLflow Artifacts
                input_example = np.random.randn(1, 3, 224, 224).astype(np.float32)

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="best_model_resnet50",
                    registered_model_name="pneumonia-resnet50",
                    serialization_format="pickle"
                )
                
        print(f"Обучение завершено. Лучший Val F1-score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
