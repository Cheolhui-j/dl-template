import torchvision.transforms as transforms

def build_transform(aug_list):
    transform_ops = []
    for aug in aug_list:
        aug_type = aug["type"]
        if aug_type == "RandomCrop":
            transform_ops.append(transforms.RandomCrop(size=aug["size"], padding=aug.get("padding", 0)))
        elif aug_type == "RandomHorizontalFlip":
            transform_ops.append(transforms.RandomHorizontalFlip())
        elif aug_type == "ToTensor":
            transform_ops.append(transforms.ToTensor())
        elif aug_type == "Normalize":
            transform_ops.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
        else:
            raise ValueError(f"Unsupported augmentation type: {aug_type}")
    return transforms.Compose(transform_ops)