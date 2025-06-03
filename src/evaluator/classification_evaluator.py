import torch
from src.evaluator.base_evaluator import BaseEvaluator
from tqdm import tqdm

class ClassificationEvaluator(BaseEvaluator):
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        pbar = tqdm(self.dataloader, desc="[Evaluator] Evaluating", unit="batch")

        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                acc = correct / total

                pbar.set_postfix(
                    {
                        "acc": f"{acc:.2f}%"
                    }
                )
                
        self.logger.info(f"Test Accuracy: {acc:.4f}")
        return acc    