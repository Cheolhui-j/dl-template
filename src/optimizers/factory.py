from torch import optim

def build_optimizer(cfg, model_params):
    name = cfg.get("name", "sgd")
    lr = cfg.get("lr", 0.1)

    if name == "sgd":
        return optim.SGD(
            model_params,
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 5e-4)
        )
    elif name == "adam":
        return optim.Adam(
            model_params,
            lr=lr,
            weight_decay=cfg.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    