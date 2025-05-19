import torch
from trainers.base_trainer import BaseTrainer

class classificationTrainer(BaseTrainer):
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
        return running_loss / len(dataloader.dataset)
    
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
            return val_loss / len(dataloader.dataset), accuracy
        
    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")