# Vision Self‑Supervised Learning (SSL) Frameworks
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-blue.svg)](https://pytorch.org/)  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Project Overview

Self‑Supervised Learning (SSL) leverages unlabeled data to learn rich visual representations. In this repository, we:
1. **Pretrain** three SSL backbones on ImageNet‑100 (unlabeled):
   - **SimCLR** (Contrastive Learning)
   - **MoCo** (Momentum Contrast)
   - **MAE** (Masked Autoencoder)  
2. **Freeze** each backbone and train a **linear classifier** on 100 labeled classes.  
3. **Compare** downstream performance in accuracy, F1 score, training efficiency, and resource usage.

---

## Installation

1. **Clone** this repository  
   ```bash
   git clone https://github.com/sarthakv162/Self-Supervised-Learning.git
   cd Self-Supervised-Learning


2. **Create & activate** a Python environment

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux / macOS
   venv\Scripts\activate.bat      # Windows
   ```

3. **Download & organize** the ImageNet‑100 subset under `data/ssl_dataset/`

   ```
   data/ssl_dataset/
   ├── train.X1/… train.X4/    # Unlabeled images for pretraining
   └── val.X/                  # Labeled images + Labels.json for linear eval
   ```

---

## Usage

### 1. Pretraining

* **SimCLR**

  ```bash
  python src/simclr/train.py \
    --data_root data/ssl_dataset \
    --train_folders train.X1 train.X2 train.X3 train.X4 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --output_dir results/simclr
  ```

* **MoCo**

  ```bash
  python src/moco/train.py \
    --data_root data/ssl_dataset \
    --train_folders train.X1 train.X2 train.X3 train.X4 \
    --batch_size 64 \
    --MOMENTUM 0.999 \
    --epochs 100 \
    --lr 1e-3 \
    --output_dir results/moco
  ```

* **MAE**

  ```bash
  python src/mae/train.py \
    --data_root data/ssl_dataset \
    --train_folders train.X1 train.X2 train.X3 train.X4 \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-4 \
    --mask_ratio 0.75 \
    --output_dir results/mae
  ```

---

### 2. Linear Evaluation

* **SimCLR Eval**

  ```bash
  python src/simclr/eval.py \
    --data_root data/ssl_dataset/val.X \
    --checkpoint results/simclr/best_model.pt \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --output_dir results/simclr/eval
  ```

* **MoCo Eval**

  ```bash
  python src/moco/eval.py \
    --data_root data/ssl_dataset/val.X \
    --checkpoint results/moco/best_model.pt \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir results/moco/eval
  ```

* **MAE Eval**

  ```bash
  python src/mae/eval.py \
    --data_root data/ssl_dataset/val.X \
    --checkpoint results/mae/mae_checkpoint_epoch50.pth \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir results/mae/eval
  ```

---

## Using Pretrained Models

### 1. Load your model and checkpoint

```python
import torch
from src.moco.model import MoCoModel  # or SimCLRModel, CustomMAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MoCoModel(base_model='resnet18', projection_dim=128)
model.encoder.fc = torch.nn.Identity()
state = torch.load('results/moco/best_model.pt', map_location=device)
model.load_state_dict(state['model_state_dict'], strict=False)
model = model.encoder.to(device)
model.eval()
```

### 2. Prepare your input image

```python
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
img = Image.open('path/to/your/image.jpg').convert('RGB')
x = preprocess(img).unsqueeze(0).to(device)
```

### 3. Extract features or classify

```python
with torch.no_grad():
    features = model(x)
print('Extracted feature vector shape:', features.shape)
```

### 4. Optional: Run a linear classifier

```python
from src.moco.eval import LinearClassifier
classifier = LinearClassifier(num_features=512, num_classes=100).to(device)
classifier.load_state_dict(torch.load('results/moco/eval/best_linear.pt'))
classifier.eval()
logits = classifier(features)
pred_class = logits.argmax(dim=-1).item()
print('Predicted class index:', pred_class)
```

---

## Key References

* **SimCLR**: Ting Chen et al., “A Simple Framework for Contrastive Learning of Visual Representations”, ICML 2020.
* **MoCo**: Kaiming He et al., “Momentum Contrast for Unsupervised Visual Representation Learning”, CVPR 2020.
* **MAE**: Kaiming He et al., “Masked Autoencoders Are Scalable Vision Learners”, CVPR 2022.
* **ViT**: Alexey Dosovitskiy et al., “An Image is Worth 16×16 Words”, ICLR 2021.
* **ResNet**: Kaiming He et al., “Deep Residual Learning for Image Recognition”, CVPR 2016.

---

## Contributing & License

* **Contributing**: Fork → branch → PR.
* **License**: MIT © 2025 Sarthak Verma. See [LICENSE](LICENSE).

---

##  Contact

Sarthak Verma
📧 [sarthak16.verma2005@gmail.com](mailto:sarthak16.verma2005@gmail.com)
🔗 GitHub: [@sarthakv162](https://github.com/sarthakv162)

Feel free to open issues or PRs with questions, feedback, or enhancements!

```

---

Let me know if you want:
-  A comparison table of results
-  Experiment tracking with TensorBoard or W&B
-  Ready-to-use Jupyter notebooks or Colab link

Happy to help fine-tune this further!
```
