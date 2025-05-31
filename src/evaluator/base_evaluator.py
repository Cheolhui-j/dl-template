from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, model, dataloader, device, logger):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger

    @abstractmethod
    def evaluate(self):
        pass
