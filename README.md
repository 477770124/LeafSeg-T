LeafSeg-TOfficial implementation of LeafSeg-T: Leaf Segmentation-Driven Drought Stress Phenotyping of Industrial Woody Crops.📁 Project StructureThe project is organized to decouple model training from downstream phenotypic analysis:PlaintextLeafSeg-T/
├── application/         # Downstream analysis modules
│   ├── compute_ldsi.py  # LDSI drought stress index calculation (Figure 17)
│   ├── lida_lesion.py   # LIDA pathology visual analysis
│   └── phenotype_ldsi.py# Morphological and spectral feature extraction
├── configs/             # Training hyperparameter configurations
├── data/                # Dataset (P-Leaf, COCO format)
├── models/              # Model architecture modules
│   ├── edge_aware_swin.py      # Backbone with edge-aware attention (M=8)
│   ├── channel_se.py           # Channel Attention Refinement
│   ├── bifpn.py                # Bidirectional Feature Pyramid Network
│   ├── spatial_cbam.py         # Spatial Attention Refinement
│   └── cascade_rcnn_head.py    # Cascade Mask R-CNN decoder
├── requirements.txt     # Environment dependencies
├── train_net.py         # Main training/evaluation entry point
└── README.md            # Project documentation
🚀 Quick Start1. InstallationBashconda create -n leafseg python=3.12 -y
conda activate leafseg
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
2. TrainingUse the provided configuration to train on the P-Leaf dataset:Bashpython train_net.py --config-file configs/leafseg_t_config.yaml --num-gpus 1
3. Phenotypic Analysis (LIDA & LDSI)After generating segmentation masks, run the application modules to extract drought indices:Bash# Compute LDSI score and generate visualization plots (Figure 17)
python application/compute_ldsi.py

# Extract disease lesion masks and visualize pathology
python application/lida_lesion.py
🔬 Key InnovationsEdge-Aware Backbone: Customizes window size to $M=8$ for fine-grained leaf texture capture.Synergistic Attention: Implements SE for channel-wise selection (pre-fusion) and CBAM for spatial-wise localization (post-fusion).Multi-Task Optimization: Balances detection, classification, and mask accuracy through cascaded IoU thresholding .  ✒️ CitationIf you use this code in your research, please cite:Code snippet@article{leafseg_t_2026,
  title={LeafSeg-T: Leaf Segmentation-Driven Drought Stress Phenotyping of Industrial Woody Crops},
  author={Ge, Tianxiao and Cai, Guoyao and Wu, Dongyang and Wu, Shifan and Xu, Sheng},
  journal={Preprint submitted to Elsevier},
  year={2026}
}
