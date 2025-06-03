# Deep Learning Project Template

A clean, modular, and extensible deep learning project template built with PyTorch. Designed for training, evaluating, and experimenting with classification tasks (extendable to others).

---

## 📁 Project Structure

```
├── configs/
│   ├── base.yaml         # All training & testing config (backbone, optimizer, scheduler, etc.)
│
├── datasets/
│   └── (external datasets managed here, gitignored)
│
├── experiments/
│   └── exp_YYYY_MM_DD_HHMM/   # Contains init.pth, best.pth, latest.pth, config backup
│
├── src/
│   ├── backbone/
│   │   └── resnet.py      # ResNet family implementation (18, 34, 50, 101, 152)
│   │
│   ├── data/
│   │   └── cifar10.py     # CIFAR10 loader with transforms
│   │
│   ├── evaluators/
│   │   ├── base_evaluator.py
│   │   └── classification_evaluator.py
│   │
│   ├── models/
│   │   └── classification.py
│   │
│   ├── optimizers/
│   │   └── factory.py     # Optimizer builder
│   │
│   ├── schedulers/
│   │   └── factory.py     # Scheduler builder
│   │
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   └── classification_trainer.py
│   │
│   └── utils/
│       └── logger.py
│
├── main.py               # Entry point for training
├── evaluate.py           # Entry point for testing
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python main.py --config configs/base.yaml
```

### 3. Evaluate

```bash
python evaluate.py --config configs/base.yaml --checkpoint experiments/exp_xxx/best.pth
```

---

## ⚙️ Key Features

* 🔧 **Backbone modifiability**: Clean ResNet support with custom block extension
* ⚡ **Training/evaluation separation** with evaluator callback
* 🧪 **Configurable with Hydra/YAML**: all training params, schedulers, and augmentations controlled from config
* 🧵 **Checkpointing**: saves init, best, latest weights in structured folders
* 🔁 **Resume training support**
* 📊 **Logging**: minimal custom logger
* 📂 **Experiments are fully tracked and reproducible**

---

## 🔄 TODO / Extensions

- [x] Design project structure and directories. (src, configs, experiments, etc.)
- [x] Implement ResNet backbone. (18, 34, 50, 101, 152)
- [x] Define classification trainer and evaluator.
- [x] Implement CIFAR10 dataset loader with data augmentation transforms with config.
- [x] Define logger utility and integrate with trainer/evaluator.
- [x] Define optimizer & scheduler and link with config.
- [x] Add checkpoint (init, best, latest checkpoints) saving/loading functionality.
- [x] Add a simple training progress UI (e.g., tqdm or custom console visualization).
- [ ] Add wandb or tensorboard support.
- [ ] Unit tests for trainer and evaluator.
- [ ] CLI with argparse or click.

---

## 📌 Notes

* To add a new model: add under `src/backbone/`, modify `build_resnet()` or add new builder
* To add a new dataset: follow the `src/data/cifar10.py` structure
* `configs/base.yaml` is the unified config entry point. You can split it later into train/test if needed.

---

## 📬 Contact (TBD)

Author: \[Your Name]
Email: \[[your\_email@example.com](mailto:your_email@example.com)]

Feel free to open issues or submit pull requests!
