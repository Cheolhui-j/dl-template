import os 
import yaml 
import torch
import torch.nn as nn
from datetime import datetime
from src.tasks.classification import Classification
from src.trainers.classification_trainer import classificationTrainer
from src.data.cifar10 import get_cifar10_loaders
from src.utils.logger import get_logger
from src.evaluator.classification_evaluator import ClassificationEvaluator
from src.schedulers.factory import build_scheduler
from src.optimizers.factory import build_optimizer

def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/exp_{timestamp}" 
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger(os.path.join(exp_dir, "train.log"))
    
    model = Classification(
        num_classes=cfg["dataset"].get("numc_classes", 10),
        backbone=cfg["backbone"].get("name", "resnet18")
    ).to(device)

    torch.save(model.state_dict(), os.path.join(exp_dir, "init.pth"))

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg["dataset"].get("batch_size", 128),
        data_dir=cfg["dataset"].get("data_dir", "./dataset/cifar10"),
        aug_config=cfg["dataset"].get("augmentation", {})
    )

    optimizer = build_optimizer(cfg["optimizer"], model.parameters())
    scheduler = build_scheduler(cfg["scheduler"], optimizer)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    def evaluator_callback(model, epoch):
        nonlocal best_acc
        evaluator = ClassificationEvaluator(model, test_loader, device, logger)
        acc = evaluator.evaluate()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best.pth"))
            logger.info(f"New best model saved with accuracy: {acc:.4f}")
        torch.save(model.state_dict(), os.path.join(exp_dir, "latest.pth"))

    trainer = classificationTrainer(model, optimizer, criterion, device, logger, scheduler=scheduler)
    trainer.train(train_loader, test_loader, epochs=cfg["backbone"].get("epochs", 20), callback=evaluator_callback)

    # 실험 결과 저장
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    main()