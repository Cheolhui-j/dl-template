import torch
from src.trainers.base_trainer import BaseTrainer

class classificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, device, logger, scheduler=None):
        super().__init__(model, optimizer, criterion, device, logger)

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.opt.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(dataloader.dataset)
        self.logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            avg_loss = val_loss / len(dataloader.dataset)
            self.logger.info(f"Val Loss: {avg_loss:.4f} | Val Accuracy: {accuracy:.4f}")
            return avg_loss, accuracy
        
    def train(self, train_loader, val_loader, epochs, callback=None, checkpoint_path=None):
        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path)

        for epoch in range(start_epoch, epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.train_one_epoch(train_loader)
            self.validate(val_loader)
            if self.scheduler:
                self.scheduler.step()
                self.logger.info(f"Scheduler Step: lr = {self.scheduler.get_last_lr()[0]:.6f}")
            if callback:
                callback(self.model, epoch)
            if checkpoint_path:
                self.save_checkpoint(epoch, checkpoint_path)