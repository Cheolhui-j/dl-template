from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def build_scheduler(cfg, optimizer):
    name = cfg.get("name", "step")
    if name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.1)
        )
    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("t_max", 50),
            eta_min=cfg.get("eta_min", 0.0)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {name}")