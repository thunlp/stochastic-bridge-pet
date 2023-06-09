# Stochastic Bridges as Effective Regularizers for Parameter-Efficient Tuning
Code for ACL 2023 Findings paper [Stochastic Bridges as Effective Regularizers for Parameter-Efficient Tuning](https://arxiv.org/abs/2305.17670).
## 0. Environment Setup
1. Create a new environment in conda:
```bash
conda create -n bridge python=3.8
conda activate bridge
```
2. Install PyTorch following the instructions on the [official website](https://pytorch.org/get-started/locally/). We use PyTorch 1.11.0.
3. Install [Apex](https://github.com/NVIDIA/apex#from-source)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
4. Install the dependencies of our codes
```bash
pip install -r requirements.txt
```

## 1. Prepare the Pre-trained Checkpoint
For the checkpoint for Megatron-BERT, you can download at [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and convert to the format that Huggingface's Transformers can load following the instructions on https://huggingface.co/nvidia/megatron-bert-uncased-345m. After Downloading and converting, place the directory of the checkpoint at `pretrained_ckpt/bert`

For the checkpoint for Deberta-V1 xlarge, you can directly download the checkpoint from Huggingface's Transformers. Place the directory of the checkpoint at `pretrained_ckpt/deberta-xlarge`

## 2. Train the Mapping $g_\gamma$
We need to train the mapping $g_\gamma$ on the pre-training corpus that the PLM was pre-trained on. Since the pre-training corpus is to large to upload, we only include some sample data in `data/pretrain_data/`. 

We have provided scripts to train $g_\gamma$ for PDF regularizer and SDE regularizer on Megatron-BERT and Deberta. The scripts are located in `scripts/fit_g`. You can run the scripts using the following command
```bash
# Commands to train g_\gamma on Megatron-BERT
bridge_type=brown_pdf GPU_NUM=8 bash scripts/fit_g/bert.sh
bridge_type=brown_sde GPU_NUM=8 bash scripts/fit_g/bert.sh
bridge_type=ou_pdf GPU_NUM=8 bash scripts/fit_g/bert.sh
bridge_type=ou_sde GPU_NUM=8 bash scripts/fit_g/bert.sh
# Commands to train g_\gamma on Deberta
bridge_type=brown_pdf GPU_NUM=8 bash scripts/fit_g/deberta.sh
bridge_type=brown_sde GPU_NUM=8 bash scripts/fit_g/deberta.sh
bridge_type=ou_pdf GPU_NUM=8 bash scripts/fit_g/deberta.sh
bridge_type=ou_sde GPU_NUM=8 bash scripts/fit_g/deberta.sh
```
The parameters of $g_\gamma$ will be saved in `checkpoint/${bridge_type}/[bert, deberta]`

## 3. Use the Mapping $g_\gamma$ to Regularize PETs on GLUEA
After training the mapping $g_\gamma$, you can use the regularizer to train the regularized PETs.
### 3.1 Download the Data
Download and extract the GLUE data according to the instructions on https://github.com/nyu-mll/GLUE-baselines, the code will generate a directory `glue_data` containing the data of GLUE. Place the directory to `data/glue_data`. 

### 3.2 Training PETs with or without Regularizer
We have provided scripts to train the PETs with or without regularizers under full-set or few-shot settings.

**Vanilla PETs**

For vanilla PETs (the original PETs without regularizer), the scripts are placed at `scripts/glue/vanilla/`. You can use the following command to train the vanilla PETs:
```bash
# Prompt tuning for Megatron-BERT on full-set GLUE
task=RTE bash scripts/glue/vanilla/glue_prompt_bert_vanilla.sh
# Prompt tuning for Megatron-BERT on few-shot GLUE
task=RTE num_shot=16 bash scripts/glue/vanilla/fewshot/glue_prompt_bert_fewshot_vanilla.sh
```
Commands for other PETs are similar.

**PETs with PDF regularizer**

For PETs with PDF regularizer, the scripts are placed at `scripts/glue/pdf/`. You can use the following command to train:
```bash
# Prompt tuning for Megatron-BERT on full-set GLUE
task=RTE bridge_type=brown_pdf bridge_weight=0.5 bash scripts/glue/pdf/glue_prompt_bert.sh
# Prompt tuning for Megatron-BERT on few-shot GLUE
task=RTE bridge_type=brown_pdf bridge_weight=0.5 bash scripts/glue/pdf/glue_prompt_bert.sh
```
There are two additional command-line arguments: `bridge_type` can be `{brown_pdf, ou_pdf}`, and `bridge_weight` is the regularization strength $\alpha$ in our paper.

**PETs with SDE regularizer**

For PETs with SDE regularizer, the scripts are placed at `scripts/glue/sde/`. You can use the following command to train:
```bash
# Prompt tuning for Megatron-BERT on full-set GLUE
task=RTE bridge_type=brown_sde bridge_weight=0.5 bash scripts/glue/pdf/glue_prompt_bert.sh
# Prompt tuning for Megatron-BERT on few-shot GLUE
task=RTE bridge_type=brown_sde bridge_weight=0.5 bash scripts/glue/pdf/glue_prompt_bert.sh
```

## 4. Citation
```bibtext
@article{chen2023stochastic,
  title={Stochastic Bridges as Effective Regularizers for Parameter-Efficient Tuning},
  author={Chen, Weize and Han, Xu and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2305.17670},
  year={2023}
}
```
