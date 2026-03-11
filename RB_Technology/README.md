# RB_Technology — Virtual Clothing Try-On

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PM2003/Gauri_RB_Tech/blob/main/RB_Technology/setup_colab.ipynb)

## Overview

RB_Technology is an AI-powered virtual clothing try-on system built on top of the ViTON (Virtual Try-On Network) architecture. It allows users to see how any clothing item would look on them without physically trying it on.

## How It Works

The system uses a 3-stage deep learning pipeline:

1. **Segmentation Generation (SegGenerator)** — Predicts the clothing segmentation map for the target outfit on the person's body
2. **Geometric Matching Module (GMM)** — Warps the clothing item to fit the person's body pose and shape using Thin Plate Spline (TPS) transformation
3. **ALIAS Generator** — Synthesizes the final photorealistic image combining the warped clothing with the person's body

## Architecture Block Diagram

```
Person Image + Pose ──────────────────────────────────────────┐
                                                              ▼
Cloth Image ──► Cloth Mask ──► SegGenerator ──► Parse Map ──► ALIASGenerator ──► Output
                    │                                ▲
                    └──────────► GMM ──► Warped Cloth
```

## Project Structure

```
RB_Technology/
├── network.py          # Neural network architectures (SegGenerator, GMM, ALIASGenerator)
├── datasets.py         # VITON dataset loader and preprocessing
├── test.py             # Inference pipeline
├── utils.py            # Helper functions
├── cloth_mask.py       # Cloth segmentation using U2NET
├── remove_bg.py        # Background removal from person images
├── run.py              # Full preprocessing + inference pipeline
├── setup_colab.ipynb   # Google Colab notebook (recommended)
└── client-side/
    ├── app.py          # Flask web server
    └── templates/
        └── index.html  # Web UI
```

## Quick Start (Google Colab — Recommended)

Click the **Open in Colab** badge above. This handles all dependencies automatically.

## Local Setup

### Requirements

- Python 3.8+
- CUDA-enabled GPU (strongly recommended)
- 8GB+ GPU VRAM

### Installation

```bash
git clone https://github.com/PM2003/Gauri_RB_Tech
cd Gauri_RB_Tech/RB_Technology
pip install -r requirements.txt
```

### Download Pretrained Checkpoints

Download the model checkpoints and place them in the `checkpoints/` folder:
- `seg_final.pth` — Segmentation model
- `gmm_final.pth` — Geometric Matching Module
- `alias_final.pth` — ALIAS Generator

### Running Inference

```bash
python test.py --name output \
  --dataset_dir ./datasets/ \
  --checkpoint_dir ./checkpoints/ \
  --save_dir ./results/
```

## Dataset Structure

```
datasets/
└── test/
    ├── image/           # Person images (768x1024)
    ├── cloth/           # Clothing item images
    ├── cloth-mask/      # Binary masks of clothing
    ├── image-parse/     # Human parsing maps
    ├── openpose-img/    # Pose visualization images
    └── openpose-json/   # Pose keypoint JSON files
```

## Running the Web App

```bash
cd client-side
python app.py
```

Then open `http://localhost:5000` in your browser.

## Citation

This project is inspired by and builds upon the VITON-HD paper:
> Choi, S., Park, S., Lee, M., & Choo, J. (2021). VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization.

## Built with ❤️ by RB_Technology
