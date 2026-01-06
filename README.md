# 🛣️ Road Damage Detection using YOLOv8

-My Submission for Crackathon 2026 by IIT Bombay

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated road damage detection and classification using YOLOv8 for the Crackathon Competition. This solution identifies and localizes five types of road damage: Longitudinal Cracks, Transverse Cracks, Alligator Cracks, Other Corruption, and Potholes.

## 📊 Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **mAP@50:95** | 0.2365 | >0.294 | 🔄 In Progress |
| **mAP@50** | 0.5003 | >0.574 | 🔄 In Progress |
| **Precision** | 0.5542 | >0.647 | 🔄 In Progress |
| **Recall** | 0.4953 | >0.520 | ✅ On Track |

## 🎯 Problem Statement

Road infrastructure maintenance is critical but manual inspection is slow and costly. This project automates the detection of road damage using computer vision to:
- Identify damage locations with bounding boxes
- Classify damage into 5 categories
- Enable proactive infrastructure maintenance

## 📁 Dataset

**RDD2022 (Road Damage Detection 2022)**
- **Training**: 26,385 labeled images
- **Validation**: 6,000 labeled images  
- **Test**: 6,000 unlabeled images
- **Total Instances**: 44,895 damage annotations

### Class Distribution

| Class ID | Damage Type | Training Instances | Percentage |
|----------|-------------|-------------------|------------|
| 0 | Longitudinal Crack | 17,807 | 39.7% |
| 1 | Transverse Crack | 8,133 | 18.1% |
| 2 | Alligator Crack | 7,224 | 16.1% |
| 3 | Other Corruption | 7,281 | 16.2% |
| 4 | Pothole | 4,450 | 9.9% |

**Dataset Links:**
- [Google Drive](https://drive.google.com/drive/folders/1JpBQ5haJCvPhD-0jUdir3GiGNbBnah93?usp=sharing)
- [Kaggle](https://www.kaggle.com/datasets/anulayakhare/crackathon-data)

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics kagglehub opencv-python matplotlib seaborn
```

### Training
```bash
python train_model.py
```

### Inference
```bash
python inference.py --model weights/best.pt --source test_images/ --output predictions/
```

## 🏗️ Model Architecture

**YOLOv8x (Extra Large)**
- **Parameters**: 68.2M
- **Input Size**: 1280×1280
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PAN (Path Aggregation Network)
- **Head**: Anchor-free decoupled detection head

### Why YOLOv8x?
- State-of-the-art accuracy for object detection
- Multi-scale feature extraction for small damage detection
- Anchor-free design for better generalization
- Efficient training with modern optimization

## 🔧 Training Configuration

```yaml
Model: YOLOv8x
Epochs: 100
Batch Size: 8
Image Size: 1280×1280
Optimizer: AdamW
Learning Rate: 0.002 → 0.001 (cosine schedule)
Weight Decay: 0.001
Warmup: 5 epochs
```

### Data Augmentation
- **Mosaic** (100%): Combines 4 images for context
- **MixUp** (20%): Blends images for robustness
- **Copy-Paste** (10%): Addresses class imbalance
- **Random Erasing** (40%): Occlusion robustness
- **HSV Augmentation**: Color variations (h=0.02, s=0.8, v=0.5)
- **Geometric**: Rotation (±15°), Translation (±20%), Scale (0.3-1.7×)
- **Horizontal Flip** (50%): Bidirectional road context

## 📈 Key Improvements

| Technique | Baseline | Enhanced | Gain |
|-----------|----------|----------|------|
| Model Size | YOLOv8m | YOLOv8x | +10-15% mAP |
| Input Resolution | 640×640 | 1280×1280 | +5-10% mAP |
| Training Epochs | 50 | 100 | +5-8% mAP |
| Augmentation | Basic | Advanced (10+) | +3-5% mAP |
| Optimizer | SGD/Auto | AdamW | +2-3% mAP |
| Multi-scale | Disabled | Enabled | +2-3% mAP |

**Expected Total Improvement**: +25-40% mAP over baseline

## 📂 Project Structure

```
road-damage-detection/
├── train_model.py          # Main training script
├── inference.py            # Inference script
├── requirements.txt        # Python dependencies
├── data.yaml              # Dataset configuration
├── README.md              # This file
├── LICENSE                # MIT License
├── weights/               # Model weights
│   └── best.pt           # Best trained model
├── predictions/           # Test predictions
│   └── *.txt             # YOLO format predictions
└── results/              # Training outputs
    ├── results.png
    └── confusion_matrix.png
```

## 🎲 Prediction Format

Each prediction file contains detections in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

Example:
```
0 0.512345 0.678901 0.123456 0.234567 0.856789
2 0.345678 0.456789 0.098765 0.112233 0.789012
4 0.234567 0.567890 0.087654 0.098765 0.745623
```

## 🖥️ Hardware Requirements

**Recommended:**
- GPU: NVIDIA T4 (16GB) or better
- RAM: 16GB+
- Storage: 50GB free space

**Training Time:**
- YOLOv8x @ 1280px: ~10-13 hours (Tesla T4)
- YOLOv8l @ 1024px: ~6-8 hours (Tesla T4)
- YOLOv8m @ 640px: ~4-5 hours (Tesla T4)

## 🔬 Validation Metrics

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@50:95 |
|-------|-----------|--------|--------|-----------|
| Longitudinal Crack | 0.534 | 0.498 | 0.478 | 0.237 |
| Transverse Crack | 0.546 | 0.41 | 0.426 | 0.176 |
| Alligator Crack | 0.596 | 0.535 | 0.559 | 0.272 |
| Other Corruption | 0.588 | 0.684 | 0.669 | 0.348 |
| Pothole | 0.506 | 0.35 | 0.37 | 0.15 |

### Overall Metrics
- **Overall mAP@50:95**: 0.2365
- **Overall mAP@50**: 0.5003
- **Average Precision**: 0.5542
- **Average Recall**: 0.4953

## 🎯 Usage Example

```python
from ultralytics import YOLO

# Load model
model = YOLO('weights/best.pt')

# Run inference
results = model.predict(
    source='test_images/',
    imgsz=1280,
    conf=0.15,
    iou=0.4,
    save_txt=True
)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: Class {cls}, Confidence: {conf:.3f}")


## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

