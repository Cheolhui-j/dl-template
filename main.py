import os 
import yaml 
import torch
import torch.nn as nn
import torch.optim as optim
from tasks.classification import Classification
from trainers.classification_trainer import classificationTrainer
from data.cifar10 import get_cifar10_loaders
from utils.logger import get_logger

def load_config():
    with open("configs/backbone.yaml", "r") as f:
        backbone_cfg = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    return {"backbone": backbone_cfg, "dataset": dataset_cfg}

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("experiments/exp001", exist_ok=True)
    logger = get_logger("experiments/exp001/train.log")
    
    model = Classification(
        num_classes=cfg["dataset"].get("numc_classes", 10),
        backbone=cfg["backbone"].get("name", "ResNet")
    ).to(device)

    train_loader, val_loader = get_cifar10_loaders(
        batch_size=cfg["dataset"].get("batch_size", 128),
        data_dir=cfg["dataset"].get("data_dir", "./dataset/cifar10")
    )

    optimizer = optim.SGD(model.parameters(), lr=cfg["backbone"].get("lr", 0.1), momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = classificationTrainer(model, optimizer, criterion, device)
    trainer.train(train_loader, val_loader, epochs=cfg["backbone"].get("epochs", 20))

    # 실험 결과 저장
    with open("experiments/exp001/config.yaml", "w") as f:
        yaml.dump(cfg, f)
    torch.save(model.state_dict(), "experiments/exp001/latest.pth")

if __name__ == "__main__":
    main()