# MTKG-Adapter

A multi-modal temporal knowledge graph (MTKG) reasoning framework that augments the
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) MLLM with a
temporal graph adapter. It fuses three modalities — **text**, **entity images**,
and **temporal KG embeddings** extracted by a frozen [RE-GCN](https://arxiv.org/abs/2104.10353)
graph encoder — and adaptively injects the fused features into the MLLM via LoRA / MoE routers /
cross-attention / learnable layer-insertion gates.

## 🔧 Quick Start
Create a virtual environment and install the required dependencies:
```
conda create -n mtkgadapter python=3.10
conda activate mtkgadapter
pip install -r requirements.txt
```
> `torch`, `torch_geometric` and `dgl` must be installed with mutually compatible versions
> (matching your CUDA toolkit).

## 📦 Model Preparation

**1. Base MLLM (Qwen3-VL-8B-Instruct)**
Download the Qwen3-VL-8B-Instruct weights and place them in `./MLLM/Qwen3-VL-8B-Instruct`
(this path is set by `--base_model`). The model is open and can be obtained from
[Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or ModelScope.

**2. Pre-trained RE-GCN graph encoder**
Download the pre-trained RE-GCN checkpoints for your dataset (GDELT / Wiki / DuEE) and place
them under `./RE-GCN_premodel/<dataset>/`. The path is set by `--peft_path` (default
`RE-GCN_premodel/GDELT`). The RE-GCN implementation is included in `./pre_trained_regcn`.

## 📊 Dataset
Supported datasets: `GDELT` (default), `Wiki`, `DuEE`.

**Download:** 
- Dataset download link: [MMTKG](https://drive.google.com/drive/folders/1EznrqYgCEYBemizo_gA3oI_AvbzKWSTl?usp=drive_link)

Place each dataset under `./dataset/<name>/` with the following layout:
```
dataset/<name>/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── test.txt
├── train_text.json
├── test_text.json
└── picture/            # entity images (optional, for the image modality)
```
The dataset directory can be overridden with the `DATASET_DIR` environment variable
(defaults to `./dataset`). Select the dataset with `--dataset <name>`.

## 🚀 Train
Hyperparameters live in `./utils/args.py` (LLM + adapter + RE-GCN + modality switches).

Single GPU:
```
python3 train.py
```

Multi-GPU (DDP, recommended — Qwen3-VL-8B is large):
```
accelerate launch --num_processes=8 train.py
```
Key switches: `--dataset`, `--base_model`, `--peft_path`, `--peft_model`, `--use_modality_image`,
`--use_modality_graph`, `--use_text_moe`, `--use_image_moe`, `--use_graph_moe`,
`--use_cross_attention`, `--add_lora`, `--num_epochs`, `--lr`.

## 🔍 Inference
To run evaluation / generate predictions:
```
python3 predict.py --peft_model checkpoint/Qwen3_8B/ --dataset GDELT
```
The script performs constrained decoding (forces a digit answer for entity ranking) and
reports Hits@1 / Hits@3 / Hits@10 and MRR.

## 🙏 Acknowledgements
This project builds on [RE-GCN](https://github.com/Lee-zix/RE-GCN)
(Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning, SIGIR 2021).


