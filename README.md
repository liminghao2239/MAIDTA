# MAIDTA
an interpretable attention-based multi-modal model for drug-target affinity prediction

## Datasets
We used two accepted benchmark datasets and one additional dataset to validate the model's generalization ability and prediction accuracy. They can be found at the following link.

- Davis: dataset/davis/original or https://www.kaggle.com/datasets/christang0002/davis-and-kiba
- KIBA: dataset/kiba/original or https://www.kaggle.com/datasets/christang0002/davis-and-kiba

## Requirements
- Python 3.8.18
- PyTorch 1.13.1
- torch-geometric 2.3+
- RDKit
- Numpy
- Pandas
- Scikit-learn
- Scipy
- tqdm

## ESM Features (Optional)
If you have pre-computed ESM-1v features as?.pt?files, place them in a directory and set?esm_dir?in?config.json. Otherwise, the model will use simplified token encoding.

## Usage

python train.py

