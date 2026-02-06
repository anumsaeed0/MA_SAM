# MA-SAM: Commands to Run

This guide provides step-by-step instructions for installation, dataset preparation, preprocessing, training, and testing of the **MA-SAM** framework for medical image segmentation.

---

## 1. Installation

Clone the repository, set up the Conda environment, and install dependencies:

```bash
# Clone the repository
git clone https://github.com/cchen-cc/MA-SAM.git
cd MA-SAM

# Create and activate Conda environment
conda create -n masam python=3.10.12
conda activate masam

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

Download raw datasets and optionally use the provided preprocessing scripts.

### BTCV Dataset

* Download from the [challenge website](https://www.synapse.org/#!Synapse:syn3379050) (registration required).
* Preprocessing script: [util_script_btcv.py](https://github.com/cchen-cc/MA-SAM/blob/main/preprocessing/util_script_btcv.py)
* Preprocessed data (optional): [Google Drive link](https://drive.google.com/file/d/1uk8cOQsX7VQBQxnwQRRtfLT-rhX4q7PD/view?usp=drive_link)

### Prostate Dataset

* Download raw data from [this link](https://liuquande.github.io/SAML/)
* Preprocessing script: [util_script_prostateMRI.py](https://github.com/cchen-cc/MA-SAM/blob/main/preprocessing/util_script_prostateMRI.py)
* Example preprocessed dataset: [Google Drive link](https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view?usp=drive_link)

### EndoVis'18 Dataset

* Download from the [challenge website](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/) (registration required)
* Preprocessing script: [util_script_endovis18.py](https://github.com/cchen-cc/MA-SAM/blob/main/preprocessing/util_script_endovis18.py)

### MSD-Pancreas Dataset

* Download from [this link](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)

**Dataset Splits:** The splits used in our experiments are provided in the [`dataset_split.md`](https://github.com/cchen-cc/MA-SAM/blob/main/preprocessing/dataset_split.md) file.

---

## 3. Preprocessing

Preprocess raw datasets to generate CSV files for training, validation, and testing:

```bash
python preprocessing/preprocess.py
```

---

## 4. Training

Before training, download the **SAM ViT_H** pretrained weights:
[SAM ViT_H weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Train the model using the following command (example with the Prostate dataset):

```bash
python MA-SAM/train.py \
    --root_path data_train/prostateD/2D_all_5slice/ \
    --output output/ \
    --ckpt sam_vit_h_4b8939.pth \
    --batch_size 1 \
    --n_gpu 1 \
    --img_size 256 \
    --use_amp
```

> Note: `trainer.py` has been modified to resolve GPU usage issues. You may compare with the original implementation if needed.

---

## 5. Testing

Test the trained model using the Prostate Site E dataset:

```bash
python test.py \
    --adapt_ckpt ../prostate_siteE_modelweights.pth \
    --data_path ../data_train/prostateD/2D_all_5slice/ \
    --ckpt ../sam_vit_h_4b8939.pth
```

**Pretrained Model for Testing:** [Google Drive link](https://drive.google.com/drive/folders/1KqbGtSp6I6M7Au4qT8cUMBFob6GMGHFi?usp=drive_link)

---
