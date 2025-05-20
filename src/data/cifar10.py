import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.data.transform import build_transform

def get_cifar10_loaders(batch_size=128, data_dir="./dataset/cifar10", aug_config=None):
    
    if aug_config is None:
        raise ValueError("augmentation config (aug_config) must be provided")

    transform_train = build_transform(aug_config.get("train", []))
    transform_test = build_transform(aug_config.get("test", []))

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader