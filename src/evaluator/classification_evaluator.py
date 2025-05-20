import torch
from src.evaluator.base_evaluator import BaseEvaluator

class ClassificationEvaluator(BaseEvaluator):
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        self.logger.info(f"Test Accuracy: {acc:.4f}")
        return acc    