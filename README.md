# LeafSeg-T

> Official PyTorch implementation of **LeafSeg-T: Leaf Segmentation-Driven Drought Stress Phenotyping of Industrial Woody Crops**  
> Submitted to *Industrial Crops and Products*

This repository contains the full source code of the LeafSeg-T framework, a high-precision leaf instance segmentation pipeline for industrial woody crops under natural field environments. Built upon an enhanced Swin Transformer backbone with synergistic attention mechanisms and cascaded instance decoding, LeafSeg-T enables accurate leaf segmentation, single-leaf phenotypic parameter extraction, disease lesion quantification, and non-destructive drought stress assessment via the proposed Leaf Drought Stress Index (LDSI).

---

##  Project Structure
```
The project is organized to decouple model training from downstream phenotypic analysis:
LeafSeg-T/
├── application/ # Downstream phenotypic analysis modules
│ ├── compute_ldsi.py # LDSI drought stress index calculation (corresponds to Figure 17)
│ ├── lida_lesion.py # LIDA module: disease lesion visualization & quantification
│ └── phenotype_ldsi.py# Morphological and spectral feature extraction pipeline
├── configs/ # Training hyperparameter configuration files
├── data/ # Dataset directory (P-Leaf dataset in COCO format)
├── models/ # Core model architecture modules
│ ├── edge_aware_swin.py # Edge-aware Swin Transformer backbone (custom window size M=8)
│ ├── channel_se.py # Channel Attention Refinement (SE module)
│ ├── bifpn.py # Bidirectional Weighted Feature Pyramid Network
│ ├── spatial_cbam.py # Spatial Attention Refinement (CBAM spatial module)
│ └── cascade_rcnn_head.py # Cascaded Mask R-CNN instance decoder
├── requirements.txt # Python environment dependencies
├── train_net.py # Main entry for model training & evaluation
└── README.md # Project documentation
```



##  Quick Start

### 1. Environment Installation
We recommend setting up the environment with Conda. The implementation is based on the Detectron2 framework.

```bash
# Create and activate virtual environment
conda create -n leafseg python=3.12 -y
conda activate leafseg

# Install basic dependencies
pip install -r requirements.txt

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
### 2. Model Training
Train LeafSeg-T on the P-Leaf dataset with the default configuration:
```bash
python train_net.py --config-file configs/leafseg_t_config.yaml --num-gpus 1
```
### 3. Downstream Phenotypic Analysis (LIDA & LDSI)
After generating segmentation masks, run the application modules for drought phenotyping:
```bash
# Calculate LDSI scores and generate visualization plots (corresponds to Figure 17)
python application/compute_ldsi.py

# Extract disease lesion masks and generate pathological visualization results
python application/lida_lesion.py
```
## Key Innovations
1.Edge-Aware Backbone: Customized Swin Transformer with window size M=8 and edge-aware position bias, which fine-grained captures leaf textures, veins and contour structures.
2.Synergistic Attention Pipeline: SE channel attention for channel-wise feature selection before fusion, paired with CBAM spatial attention for spatial-wise foreground localization after fusion, effectively suppressing complex background interference.
3.Cascaded Multi-Task Optimization: Progressive instance refinement via cascaded IoU thresholds, balancing detection, classification and mask accuracy for dense overlapping leaves.

## Data Availability
The P-Leaf dataset used in this study is available from the corresponding author upon reasonable request.
