# An Automatic End-to-End Framework for Continuous Ice–Bed Interface Extraction from Ice-Penetrating Radar Data via Frequency–Spatial Deep Learning


# 🧊 PickFormer  
### An Automatic End-to-End Framework for Continuous Ice–Bed Interface Extraction  
### via Frequency–Spatial Deep Learning  

[![Python](https://img.shields.io/badge/Python-3.9-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()
[![Paper](https://img.shields.io/badge/Paper-JSTARS-blueviolet)]()

</div>

---

## 🌍 Overview

**PickFormer** is a physically informed, end-to-end deep learning framework for automatic and continuous extraction of the **ice–bed interface** from airborne ice-penetrating radar (IPR) data.

Accurate interface detection is essential for:

- Ice thickness estimation  
- Subglacial geomorphology analysis  
- Antarctic ice-sheet stability assessment  
- Ice-sheet mass balance modeling  

Traditional methods suffer from:

- ❌ Interface discontinuities  
- ❌ Weak bed-return ambiguity  
- ❌ Noise contamination and clutter interference  
- ❌ Limited cross-region generalization  

PickFormer addresses these challenges through a **frequency–spatial transformer architecture**, explicitly integrating spectral discriminative features with spatial structural modeling.

---

## 🧠 Framework Overview

<div align="center">
<img src="Figure/Figure1.png" width="85%">
</div>

**Core Components:**

- 🔹 CNN Backbone Encoder  
- 🔹 G Module (Global Spatial Modeling)  
- 🔹 F Module (Frequency–Spatial Attention)  
- 🔹 Multi-scale Decoder  

---

## 🖼 Visual Results

<div align="center">
<img src="assets/visual_example.png" width="85%">
</div>

- Continuous bed extraction under weak reflection  
- Strong clutter suppression  
- Robust cross-region performance  

---

## 🛰 Dataset

Validated on airborne IPR datasets from:

- AGAP  
- Totten Glacier  
- Pine Island Glacier  
- Antarctic Peninsula  

### Data Format

- Input: Normalized radar amplitude  
- Label: Binary ice–bed interface mask  
- Format: `.h5` / `.npy`  
- Patch-based slice training  

### Data Split

```
split/
├── train.json
├── val.json
└── test.json
```

> Raw radar data follow data usage policies (Operation IceBridge, CHINARE).

---

## 📁 Project Structure

```
PickFormer/
│
├── models/
│   ├── pickformer.py
│   ├── modules_g.py
│   ├── modules_f.py
│   └── backbone/
│
├── datasets/
├── training/
│   ├── train.py
│   ├── loss.py
│   └── metrics.py
│
├── inference/
│   ├── test.py
│   └── visualize.py
│
├── configs/
├── checkpoints/
├── split/
└── assets/
```

---

## 🚀 Installation

```bash
git clone https://github.com/vivian-ma97/PickFormer.git
cd PickFormer

conda create -n pickformer python=3.9
conda activate pickformer

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python training/train.py \
    --config configs/pickformer.yaml \
    --split split/train.json
```

Metrics:

- mIoU  
- Dice  
- F1-score  
- Kappa  
- MAE (optional regression mode)

---

## 🔍 Inference

```bash
python inference/test.py \
    --model checkpoints/pickformer_best.pth \
    --split split/test.json
```

Outputs:

```
demo_test_pred/
├── pred/
└── gt/
```

---

## 📦 Pretrained Weights

Download:

```
https://your-weight-link-here
```

Place into:

```
checkpoints/
```

---

## 🔬 Reproducibility

To reproduce paper results:

1. Download pretrained weights  
2. Use provided test split  
3. Run inference  
4. Metrics computed automatically  

All hyperparameters are in:

```
configs/
```

---

## 📊 Performance

PickFormer consistently outperforms:

- U-Net  
- U-Net + ASPP  
- CNN-only baselines  

Especially under:

- Weak bed-return  
- Strong clutter  
- Complex basal terrain  

---

## 📖 Citation

```bibtex
@article{ma2026pickformer,
  title={An Automatic End-to-End Framework for Continuous Ice–Bed Interface Extraction from Ice-Penetrating Radar Data via Frequency–Spatial Deep Learning},
  author={Ma, Qian and ...},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026}
}
```

---

## 🔓 Code Availability

Source code:

https://github.com/vivian-ma97/PickFormer  

The full implementation will be uploaded within **7 days after manuscript submission**.

---

## 📜 License

MIT License
