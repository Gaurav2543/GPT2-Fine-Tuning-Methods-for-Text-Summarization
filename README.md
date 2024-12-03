# GPT-2 Fine-tuning Methods for Text Summarization: A Comparative Study

## Overview
This project implements and compares three different fine-tuning approaches for the GPT-2 language model on the text summarization task using the CNN/Daily Mail dataset. The implemented methods are:
- Prompt Tuning
- LoRA (Low-Rank Adaptation)
- Traditional Fine-tuning

## Features
- Implementation of three state-of-the-art fine-tuning methods
- Comprehensive evaluation metrics including ROUGE scores
- Resource utilization tracking (GPU/CPU usage)
- Visualization of training metrics
- Parameter-efficient training approaches

## Requirements
```python
torch
transformers
datasets
rouge_score
numpy
pandas
matplotlib
seaborn
tqdm
psutil
GPUtil
```

## Project Structure
```
.
├── src/
│   ├── pt.py        # Prompt Tuning implementation
│   ├── lora.py      # LoRA implementation
│   └── ft.py        # Traditional Fine-tuning implementation
├── training_metrics/ # Training results and visualizations
├── Report.pdf
└── README.md
```

## Installation
```bash
git clone https://github.com/Gaurav2543/gpt2-fine-tuning-methods.git
cd gpt2-fine-tuning-methods
```

## Usage
Each fine-tuning method can be run separately:

```bash
# For Prompt Tuning
python src/pt.py

# For LoRA
python src/lora.py

# For Traditional Fine-tuning
python src/ft.py
```

## Model Configurations

### Prompt Tuning
- Soft prompt length: 20 tokens
- Random initialization range: ±0.5
- Only prompt embeddings trainable

### LoRA
- Rank: 8
- Alpha: 32
- Applied to attention layers
- Zero initialization for B matrices
- Normal initialization for A matrices

### Traditional Fine-tuning
- Only last transformer block and LM head trainable
- All other parameters frozen

## Common Training Parameters
- Base Model: GPT-2 Small
- Number of Epochs: 20
- Batch Size: 8
- Initial Learning Rate: 1e-4
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Learning Rate Schedule: Linear with warmup (100 steps)
- Dataset: 10% of CNN/Daily Mail dataset
- Maximum sequence lengths: 512 (input), 128 (summary)

## Results
The project compares the three methods across several metrics:
- Training and Validation Loss
- ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
- GPU/CPU Usage
- Training Time
- Parameter Efficiency

## Acknowledgments
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
