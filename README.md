# BISINDO Emergency Sign Detection System

Sistem Deteksi Isyarat Darurat BISINDO Berbasis Skeleton-Graph Menggunakan Arsitektur TCN yang Tahan Terhadap Variasi Pencahayaan dan Oklusi Parsial.

## Overview

Sistem real-time untuk mendeteksi 10 isyarat darurat dalam Bahasa Isyarat Indonesia (BISINDO) menggunakan:
- **MediaPipe Holistic** untuk ekstraksi skeleton
- **Skeleton-Graph Encoding** untuk representasi spasial
- **Temporal Convolutional Network (TCN)** untuk pemodelan temporal
- **PyTorch** dengan akselerasi MPS (Apple Silicon)

## Project Structure

```
BISINDO/
├── config/           # Configuration files
├── data/             # Dataset directory
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── models/       # Model architectures
│   ├── training/     # Training utilities
│   ├── inference/    # Inference pipeline
│   └── utils/        # Helper utilities
├── scripts/          # Executable scripts
├── notebooks/        # Jupyter notebooks
├── checkpoints/      # Model checkpoints
├── logs/             # Training logs
└── tests/            # Unit tests
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Collection

```bash
# Start recording session
python scripts/record_data.py --subject S01 --class TOLONG
```

### Training

```bash
# Train model
python scripts/train.py --config config/default.yaml
```

### Demo

```bash
# Run real-time demo
python scripts/demo.py --model checkpoints/best_model.pt
```

## Dataset

| Parameter | Value |
|-----------|-------|
| Classes | 10 emergency signs |
| Subjects | 12 |
| Reps/Subject | 15 |
| Lighting | 3 conditions |
| Occlusion | 2 conditions |
| Total Samples | ~10,800 |

### Classes

1. TOLONG - Help request
2. BAHAYA - Danger warning
3. KEBAKARAN - Fire
4. SAKIT - Sick/Medical emergency
5. GEMPA - Earthquake
6. BANJIR - Flood
7. PENCURI - Thief/Robber
8. PINGSAN - Unconscious
9. KECELAKAAN - Accident
10. DARURAT - General emergency

## Architecture

```
Input Video → MediaPipe → Keypoints → Graph Encoder → TCN → Attention → Classification
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy (normal) | ≥ 90% |
| Accuracy (low light) | ≥ 80% |
| Accuracy (occluded) | ≥ 75% |
| Inference FPS | ≥ 15 |
| Model Size | ≤ 200 MB |

## License

This project is for educational purposes (Science Fair Project).
But everyone is free to use it. Please credit me if you use it.
[MIT License](LICENSE.md)

## Author

Gung Wah (Akiro Kazuki)