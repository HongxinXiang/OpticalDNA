# OpticalDNA: Rethinking Genomic Modeling Through Optical Character Recognition

Official code release for **OpticalDNA: Rethinking Genomic Modeling Through Optical Character Recognition**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
<a href="https://github.com/HongxinXiang/OpticalDNA/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/HongxinXiang/OpticalDNA?style=flat-square"></a>
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HongxinXiang/OpticalDNA?style=flat-square">
<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=HongxinXiang.OpticalDNA-X&left_color=gray&right_color=orange"></a>


<a href='https://hongxinxiang.github.io/projects/OpticalDNA/'><img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=rescript&logoColor=pink'></a>
<a href='https://arxiv.org/abs/2602.02014'><img src='https://img.shields.io/badge/Arxiv-2602.02014-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://arxiv.org/pdf/2602.02014'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>

---

## Project Directory / Table of Contents

- [📢 News](#-news)
- [📁 Repository Structure](#-repository-structure)
- [🧪 1. Summary](#-1-summary)
  - [1.1 Highlights](#11-highlights)
  - [1.2 Citation](#12-citation)
- [⚙️ 2. Environment](#%EF%B8%8F-2-environment)
  - [2.1 GPU Environment](#21-gpu-environment)
  - [2.2 Conda Environment Setup](#22-conda-environment-setup)
- [🧬 3. Install VisualDNA](#-3-install-visualdna)
- [🗂️ 4. Data Preparation](#%EF%B8%8F-4-data-preparation)
  - [4.1 Data Layout](#41-data-layout)
  - [4.2 Raw Data Download](#42-raw-data-download)
  - [4.3 Process Raw Data with VisualDNA](#43-process-raw-data-with-visualdna)
  - [4.4 Command Examples](#44-command-examples)
  - [4.5 Notes](#45-notes)
- [🚀 5. Pre-training](#-5-pre-training)
  - [5.1 HG38 Example](#51-hg38-example)
  - [5.2 Rice Example](#52-rice-example)
- [📝 6. Notes for Release Users](#-6-notes-for-release-users)
- [📄 7. License](#-7-license)

## 📢 News

- **[2025/05/09]** Repository initialized!

- **[2026/04/30]** 🎉 Paper was accepted by _ICML 2026_ !

---


## 📁 Repository Structure

```text
OpticalDNA/
├── opticaldna/                 # 1. Model configuration, tokenizer, and OpticalDNA model wrappers
├── src/                        # 2. Training and data-loading source code
│   ├── pretrain_opticaldna.py   #    Pre-training entry point
│   └── opticaldna_data/         #    Dataset, conversation builder, and data collator
├── scripts/                    # 3. Standalone utility scripts
│   └── data/                   #    Data preparation utilities
├── assets/                     # 4. Figures used by the README
├── environment.yml             # 5. Conda environment specification
├── LICENSE
└── README.md
```

The main directories are:

1. `opticaldna/`: model configuration, tokenizer files, and OpticalDNA model wrappers.
2. `src/`: training entry point and data-loading modules used during pre-training.
3. `scripts/data/`: standalone data preparation scripts for generating processed VisualDNA data.
4. `assets/`: figures used in this README.

## 🧪 1. Summary

OpticalDNA reformulates genomic sequence modeling as a document-understanding problem. DNA sequences are rendered into structured visual pages, encoded by a vision-language backbone, and trained with prompt-conditioned genomic objectives including reading, grounding, ROI transcription, masked completion, subsequence localization, and chromosome-level recognition.


<p align="center">
  <img src="assets/framework.png" width="800">
</p>

### 1.1 Highlights

- **DNA as visual documents.** Long genomic sequences are converted into multi-page DNA documents with pixel-level coordinate annotations.
- **Prompt-conditioned genomic pretraining.** Six OCR-style tasks cover recognition, grounding, retrieval, and completion.
- **Efficient long-context representation.** Visual tokens provide compact representations for downstream long-range genomic prediction.
- **Lightweight downstream adaptation.** The pretrained visual encoder can be reused with linear or shallow MLP heads.


### 1.2 Citation

> 📄 **Citation**
>
> ```bibtex
> @inproceedings{xiang2026rethinking,
>   title         = {Rethinking Genomic Modeling Through Optical Character Recognition},
>   author        = {Xiang, Hongxin and Ma, Pengsen and Cao, Yunkang and Yu, Di and Chen, Haowen and Yang, Xinyu and Zeng, Xiangxiang},
>   journal       = {arXiv preprint arXiv:2602.02014},
>   year          = {2026},
>   url           = {https://arxiv.org/abs/2602.02014}
> }
> ```

---

## ⚙️ 2. Environment

### 2.1 GPU Environment

- CUDA Toolkit: 11.8
- CUDA: 7.5
- Triton: 3.4.0
- Unsloth 2025.10.12
- Transformers: 4.56.2
- Torch: 2.6.0+cu118

### 2.2 Conda Environment Setup

Install the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate opticaldna
```

Alternatively, you can create the environment manually:

```bash
conda create -n opticaldna python=3.12 -y
conda activate opticaldna

pip install PyMuPDF img2pdf einops easydict addict Pillow
pip install flash-attn==2.7.3 --no-build-isolation --use-pep517 --no-deps

pip install pytabix pandas pyfaidx
pip install kipoiseq==0.5.2
pip install tensorflow==2.16.1
pip install selene-sdk==0.5.3  # Requires torch <= 2.3.1
pip install natsort==8.4.0
pip install pyBigWig==0.3.23
pip install decorator
pip install rdkit
pip install dictionary omegaconf safetensors
pip install timm==1.0.22 --no-deps
pip install tf-keras

pip install datasets==4.3.0
pip install trl==0.24.0 --no-deps
pip install transformers==4.56.2 --no-deps
pip install peft==0.17.1
pip install tokenizers==0.22.1
pip install "unsloth[cu118-torch260]"

# Optional: install PyTorch and related local wheels if needed
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# download from https://pytorch-geometric.com/whl/torch-2.6.0%2Bcu118.html
pip install torch_cluster-1.6.3+pt26cu118-cp312-cp312-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt26cu118-cp312-cp312-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt26cu118-cp312-cp312-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt26cu118-cp312-cp312-linux_x86_64.whl

pip install numpy==1.26
```

---

## 🧬 3. Install VisualDNA

```bash
pip install visualdna
```

## 🗂️ 4. Data Preparation

### 4.1 Data Layout

OpticalDNA uses data generated by the VisualDNA rendering pipeline. Each dataset contains a `raw/` directory for source files and a `processed/` directory for rendered DNA document images, bounding boxes, and metadata.

A typical dataset directory is organized as follows:

```text
/path/to/opticaldna_dataset/
└── <dataset_name>/
    ├── raw/
    │   ├── <dataset_name>.parquet      # Source genome files, annotations, or metadata
    │   └── ...
    └── processed/
        └── <render_config_name>/
            ├── index.csv              # Metadata index used by OpticalDNA
            ├── images/                # Rendered DNA document images
            └── bbox/                  # Bounding-box annotations
```

In the training scripts, set `--dataroot` to the parent data directory and `--dataset` to the dataset subdirectory name.

For example, if the dataset is stored as:

```text
/path/to/opticaldna_dataset/hg38-2048/
```

then the corresponding arguments should be:

```bash
--dataroot /path/to/opticaldna_dataset
--dataset hg38-2048
```

### 4.2 Raw Data Download

We provide the raw data links used to construct the OpticalDNA pre-training datasets. After downloading the raw files, place them under the corresponding `raw/` directory before running the VisualDNA rendering pipeline.

| Dataset | Description | Raw Data Link | Expected Raw Directory |
|---|---|---|---|
| HG38 | Human reference genome data used for OpticalDNA pre-training | [Download](<HG38_RAW_DOWNLOAD_LINK>) | `/path/to/opticaldna_dataset/hg38-2048/raw/` |
| Rice | Rice genome data used for OpticalDNA pre-training | [Download](<RICE_RAW_DOWNLOAD_LINK>) | `/path/to/opticaldna_dataset/rice/raw/` |

### 4.3 Process Raw Data with VisualDNA

The data processing utilities are placed under:

```text
scripts/
└── data/
    ├── generate_processed.py
    └── add_raw_columns_to_processed_index.py
```

The processing workflow contains two steps:

1. Generate the `processed/` directory from raw genome files.
2. Add selected metadata columns from the raw file to the generated `processed/index.csv`.

**Step 1: Generate processed data**

Run `scripts/data/generate_processed.py` to convert raw genome sequences into rendered DNA document images and bounding-box annotations.

#### 5.1 HG38 Example

```bash
python scripts/data/generate_processed.py \
  --dataroot /path/to/opticaldna_dataset \
  --dataset hg38-2048 \
  --raw-format parquet \
  --seq-columns seq \
  --img-width 640 \
  --img-height 640 \
  --font-size 14 \
  --line-spacing 1.6 \
  --merge-pages \
  --save-bbox \
  --shard-size auto
```

#### 5.2 Rice Example

```bash
python scripts/data/generate_processed.py \
  --dataroot /path/to/opticaldna_dataset \
  --dataset rice \
  --raw-format parquet \
  --seq-columns seq \
  --img-width 640 \
  --img-height 640 \
  --font-size 14 \
  --line-spacing 1.6 \
  --merge-pages \
  --save-bbox \
  --shard-size auto
```

The command above corresponds to the following VisualDNA configuration:

```python
from visualdna.data import ShardedBuilder
from visualdna.render import BaseRenderConfig

config = BaseRenderConfig(
    img_width=640,
    img_height=640,
    font_size=14,
    line_spacing=1.6,
    merge_pages=True,
    save_bbox=True,
)

builder = ShardedBuilder(
    dataroot="/path/to/opticaldna_dataset",
    dataset="hg38-2048",
    render_config=config,
    seq_columns=["seq"],
    raw_csv_url=None,
    force_generate=False,
    shard_size="auto",
    raw_format="parquet",
)
```

After this step, the processed files should be generated under:

```text
/path/to/opticaldna_dataset/
└── hg38-2048/
    ├── raw/
    └── processed/
        └── <render_config_name>/
            ├── index.csv
            ├── images/
            └── bbox/
```

**Step 2: Add metadata columns to `processed/index.csv`**

Some downstream tasks require extra metadata columns, such as chromosome names. These columns can be copied from the raw file into the generated `processed/index.csv`.

Run `scripts/data/add_raw_columns_to_processed_index.py` after Step 1.

**Add `chr_name` for HG38**

```bash
python scripts/data/add_raw_columns_to_processed_index.py \
  --dataroot /path/to/opticaldna_dataset \
  --processed-dataset hg38-2048 \
  --raw-dataset hg38-2048 \
  --render-id <render_config_name> \
  --raw-format parquet \
  --key index \
  --columns chr_name
```

**Add multiple columns**

```bash
python scripts/data/add_raw_columns_to_processed_index.py \
  --dataroot /path/to/opticaldna_dataset \
  --processed-dataset hg38-2048 \
  --raw-dataset hg38-2048 \
  --render-id <render_config_name> \
  --raw-format parquet \
  --key index \
  --columns chr_name split species
```

Here:

- `--processed-dataset` is the dataset whose `processed/index.csv` will be updated.
- `--raw-dataset` is the dataset where the source raw file is stored.
- `--render-id` is the generated render configuration directory name under `processed/`.
- `--key` is the column used to match rows between raw data and processed data.
- `--columns` specifies one or more raw metadata columns to add to `processed/index.csv`.

The expected processed index path is:

```text
/path/to/opticaldna_dataset/<processed-dataset>/processed/<render-id>/index.csv
```

The expected raw file path is:

```text
/path/to/opticaldna_dataset/<raw-dataset>/raw/<raw-dataset>.parquet
```

### 4.4 Command Examples

Most pre-training workflows are expected to run on Linux servers. Windows `cmd` examples are also provided for users who prepare data locally.

**Linux/macOS examples**

**Generate processed data**

```bash
python scripts/data/generate_processed.py \
  --dataroot /path/to/opticaldna_dataset \
  --dataset hg38-2048 \
  --raw-format parquet \
  --seq-columns seq \
  --img-width 640 \
  --img-height 640 \
  --font-size 14 \
  --line-spacing 1.6 \
  --merge-pages \
  --save-bbox \
  --shard-size auto
```

**Add metadata columns**

```bash
python scripts/data/add_raw_columns_to_processed_index.py \
  --dataroot /path/to/opticaldna_dataset \
  --processed-dataset hg38-2048 \
  --raw-dataset hg38-2048 \
  --render-id render_w640_h640_fs14_ls1.6_hash_f345fcfc \
  --raw-format parquet \
  --key index \
  --columns chr_name
```

**Windows CMD examples**

If you run the commands in Windows `cmd`, use `^` for line continuation.

**Generate processed data**

```cmd
python scripts\data\generate_processed.py ^
  --dataroot F:\path\to\opticaldna_dataset ^
  --dataset hg38-2048 ^
  --raw-format parquet ^
  --seq-columns seq ^
  --img-width 640 ^
  --img-height 640 ^
  --font-size 14 ^
  --line-spacing 1.6 ^
  --merge-pages ^
  --save-bbox ^
  --shard-size auto
```

**Add metadata columns**

```cmd
python scripts\data\add_raw_columns_to_processed_index.py ^
  --dataroot F:\path\to\opticaldna_dataset ^
  --processed-dataset hg38-2048 ^
  --raw-dataset hg38-2048 ^
  --render-id render_w640_h640_fs14_ls1.6_hash_f345fcfc ^
  --raw-format parquet ^
  --key index ^
  --columns chr_name
```

### 4.5 Notes

- `raw/{dataset}.parquet` should exist before running `generate_processed.py`.
- The `processed/` directory is generated automatically by VisualDNA.
- The `render-id` should match the directory name generated under `processed/`.
- The same scripts can be used for HG38, rice, or any other dataset by changing parser arguments.
- Keep these scripts under `scripts/data/` because they are command-line data preparation utilities rather than model or training modules.


## 🚀 5. Pre-training

OpticalDNA is initialized from the DeepSeek-OCR checkpoint. Download the checkpoint from [this link](https://1drv.ms/u/c/53030532e7d1aed6/IQCG1Ja7O8kUQo1_OAPl-1EDAa9pzp4oNGJl5JWr0Tr-yZQ?e=Wr4Hpe) and place the extracted checkpoint directory under `opticaldna/`.


### 5.1 HG38 Example

```bash
conda activate opticaldna

model_name=./opticaldna
output_dir=./outputs/pretrain_opticaldna/hg38
dataroot=/path/to/opticaldna_data
dataset=hg38-2048

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTHONUNBUFFERED=1 \
python -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29501 \
src/pretrain_opticaldna.py --backend nccl \
  --dataloader_num_workers 12 --dataloader_prefetch_factor 4 \
  --per_device_train_batch_size 8 --save_total_limit 10 --save_steps 1000 \
  --warmup_steps 20000 --output_dir ${output_dir} \
  --dataroot ${dataroot} --dataset ${dataset} \
  --model_name ${model_name} \
  --task_sampling "p_t1=0.17,p_t2=0.17,p_t3=0.17,p_t4=0.17,p_t5=0.17,p_t6=0.15" \
  --tail_truncation "enabled=true,base_delete_ratio=0,max_delete_ratio=0.98" \
  --line_span_cfg "min_n_base=1,max_n_base=8,min_n_sample=1,max_n_sample=3,unique_lines=true" \
  --subseq_locate_cfg "min_len=6,max_len=32,allow_overlap=true" \
  --lora_r 128 --trainable_modules_to_save page_fusion_layer \
  --learning_rate 5e-4 --max_steps 281250 --annealed_sampler_total_steps 1 \
  --lora_projector
```

### 5.2 Rice Example

```bash
conda activate opticaldna

model_name=./opticaldna
output_dir=./outputs/pretrain_opticaldna/rice
dataroot=/path/to/opticaldna_data
dataset=rice-2048  # we use NIP-T2T_w2048_o1920_SeqCase.UPPER
max_steps=235405

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTHONUNBUFFERED=1 \
python -m torch.distributed.run --standalone --nproc_per_node=8 --master_port=29501 \
src/pretrain_opticaldna.py --backend nccl \
  --dataloader_num_workers 12 --dataloader_prefetch_factor 4 \
  --per_device_train_batch_size 8 --save_total_limit 10 \
  --warmup_steps 20000 --output_dir ${output_dir} \
  --dataroot ${dataroot} --dataset ${dataset} \
  --model_name ${model_name} \
  --task_sampling "p_t1=0.17,p_t2=0.17,p_t3=0.17,p_t4=0.17,p_t5=0.17,p_t6=0.15" \
  --tail_truncation "enabled=true,base_delete_ratio=0,max_delete_ratio=0.9" \
  --line_span_cfg "min_n_base=1,max_n_base=8,min_n_sample=1,max_n_sample=3,unique_lines=true" \
  --subseq_locate_cfg "min_len=6,max_len=32,allow_overlap=true" \
  --lora_r 128 --trainable_modules_to_save page_fusion_layer \
  --learning_rate 5e-4 --max_steps ${max_steps} --lora_projector --lora_decoder
```

## 📝 6. Notes for Release Users

- Replace all example paths with local paths before running.
- Use `HF_HUB_OFFLINE=1` when loading only local checkpoints.
- Large model weights and generated datasets are intentionally excluded from the repository. Put them under local paths and pass them through command-line arguments.
- The model code is adapted for OpticalDNA while preserving compatibility with the underlying OCR-style vision-language architecture.


## 📄 7. License

This repository is released under the MIT License. Parts of the model implementation are adapted from third-party open-source projects; please also follow the corresponding upstream licenses and notices where applicable.
