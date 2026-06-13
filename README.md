# CrossGAN-Omics

CrossGAN-Omics is a class-aware generative framework for bidirectional translation between DNA methylation and gene expression in latent space. It is designed to address the challenge of limited paired multi-omic data by learning cross-modal relationships between transcriptomic and epigenomic representations.

The framework operates in two stages. First, each modality is compressed into a low-dimensional latent representation using autoencoders. Second, a conditional CycleGAN-based model learns bidirectional mappings between these latent spaces, conditioned on class labels. This enables generation of synthetic samples in either modality while preserving class-specific biological structure.

<p align="center">
  <img src="images/framework.png" width="800">
</p>

---

## Installation

Create and activate the conda environment:

```bash
conda env create --prefix ./env --file environment.yml --force
conda activate ./env
```

---

## Usage

### Step 1: Train Autoencoders and Generate Latent Representations

Run the autoencoder script to compress both modalities into a 50-dimensional latent space:

```bash
python autoencoder_train.py \
  --input_file1 <expression_data.csv> \
  --input_file2 <methylation_data.csv> \
  --output_file1 <expression_latent.csv> \
  --output_file2 <methylation_latent.csv> \
  --dropout_rate 0.2 \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --epochs 200
```

Input format:
- CSV files
- Rows represent samples
- Columns represent features
- The last column must be the condition label

Output:
- Two CSV files containing 50-dimensional latent representations for each modality

---

### Step 2: Train CrossGAN-Omics

Run the CrossGAN training script using the latent representations:

```bash
python crossGAN_train.py \
  --data_path <path_to_latent_data> \
  --train_file <latent_A.csv> <latent_B.csv> \
  --test_file <latent_A_test.csv> <latent_B_test.csv> \
  --batch_size 16 \
  --epochs 400
```

This step trains the conditional CycleGAN model to learn bidirectional mappings between modalities and generate synthetic latent samples.

---

## Notes

- Latent representations generated from autoencoder_train.py are required before training the GAN
- Condition labels are used for class-aware generation
- Generated latent samples can be used for downstream tasks such as data augmentation or predictive modeling
