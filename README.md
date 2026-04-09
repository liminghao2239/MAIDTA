# MDNN-DTA
A multimodal deep neural network for drug-target affinity prediction

## Datasets
We used two accepted benchmark datasets and one additional dataset to validate the model's generalization ability and prediction accuracy. They can be found at the following link.

- Davis: dataset/davis/original or https://www.kaggle.com/datasets/christang0002/davis-and-kiba
- KIBA: dataset/kiba/original or https://www.kaggle.com/datasets/christang0002/davis-and-kiba

## Requirements
- Python 3.8.18
- PyTorch 1.13.1
- Biopython
- Numpy
- Pandas
- Pillow
- Rdkit-pypi
- Scikit-learn
- Scipy
- Fair-esm

## Training
We need to extract the protein feature from the ESM first.

python run_esm.py \
--model-location "esm1v_t33_650M_UR90S_1" \
--fasta-file dataset/davis/original/davisFASTA.fasta \
--output-dir dataset/protein_feature/

Then train the model:

python train.py --config default_config.json
