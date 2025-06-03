from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, optimizer, criterion, device, logger):
        self.model = model
        self.opt = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger

        self.current_epoch = 0
        self.total_epochs = 0

    @abstractmethod
    def train_one_epoch(self, dataloader):
        pass

    @abstractmethod
    def validate(self, dataloader):
        pass

    @abstractmethod
    def train(self, train_loader, val_loader, epochs):
        pass

