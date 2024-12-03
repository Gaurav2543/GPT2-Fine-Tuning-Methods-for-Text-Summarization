# Fine-tuning Methods Comparison

## Overview

This repository contains the implementation of three different fine-tuning methods for the GPT-2 small model on the summarization task:

1. Prompt Tuning (pt.py)
2. LoRA - Low-Rank Adaptation (lora.py)
3. Traditional Fine-Tuning (Last Layers Only) (ft.py)

## File Structure

```
GPT2-Fine-Tuning-Methods-for-Text-Summarization
│
├── README.md
├── Report.pdf
├── ANLP_Assignment_3.pdf  # Description of the Task
│
├── src/
│   ├── pt.py              # Prompt Tuning implementation
│   ├── lora.py            # LoRA implementation
│   └── ft.py              # Traditional Fine-tuning implementation
│
└── training_metrics       # Training results and visualizations
    ├── prompt-tuning /             
    ├── lora/            
    └── fine-tuning/ 
```

## Dependencies

```bash
pip install torch transformers datasets rouge-score pandas numpy seaborn matplotlib tqdm psutil gputil
```

## Dataset Setup

1. Download the CNN/Daily Mail dataset from Kaggle:
   https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/code
2. Place the dataset files in the following structure:

```
path/to/gowrishankarp/newspaper-text-summarization-cnn-dailymail/
├── train.csv
├── validation.csv
└── test.csv
```

3. Update the `path` variable in each script to match your dataset location:

```python
path = 'path/to/gowrishankarp/newspaper-text-summarization-cnn-dailymail'
```

## Model Checkpoints

The best model checkpoints for each method are automatically saved during training:

- `best_prompt_tuning_model.pt`
- `best_lora_model.pt`
- `best_traditional_finetuning_model.pt`
  
Due to file size limitations, the models are available at the following link: [Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/gaurav_bhole_research_iiit_ac_in/Eg01MitWZihCvncBK5-kCooBNj8iaWBog4AGnmjPxjVwqw?e=gsvL9E)

## Running the Code

Each implementation can be run independently:

### 1. Prompt Tuning

```bash
python src/pt.py
```

### 2. LoRA

```bash
python src/lora.py
```

### 3. Traditional Fine-tuning

```bash
python src/ft.py
```

## Restoring Pre-trained Models

To load a saved model checkpoint:

```python
# For Prompt Tuning
model = GPT2PromptTuningModel.from_pretrained('gpt2')
model.load_state_dict(torch.load('best_prompt_tuning_model.pt'))

# For LoRA
model = GPT2LoRAModel.from_pretrained('gpt2', rank=8, alpha=32)
model.load_state_dict(torch.load('best_lora_model.pt'))

# For Traditional Fine-tuning
model = GPT2TraditionalFineTuningModel.from_pretrained('gpt2')
model.load_state_dict(torch.load('best_traditional_finetuning_model.pt'))
```

## Training Metrics

Each implementation saves training metrics in the `training_metrics` directory:

- Training curves plot (PNG)
- Detailed metrics (CSV)
- Training report (TXT)
- Complete metrics data (JSON)

## Implementation Details

### Common Features

- Batch size: 8
- Maximum epochs: 20
- Learning rate: 1e-4
- Dataset: 10% of CNN/Daily Mail
- Early stopping based on validation loss
- Gradient clipping with norm 1.0
- AdamW optimizer with weight decay 0.01
- Linear learning rate scheduler with warmup

### Model-specific Details

1. **Prompt Tuning**

   - Trainable soft prompt embeddings (20 tokens)
   - Frozen transformer and LM head
   - Only soft prompt parameters are updated
2. **LoRA**

   - Rank: 8
   - Alpha: 32
   - Applied to attention layers (query, key, value, and output projections)
   - Original model parameters are frozen
3. **Traditional Fine-tuning**

   - Only last transformer block and LM head are trainable
   - All other parameters are frozen

## Notes

- GPU memory usage is monitored and reported
- ROUGE scores are calculated every epoch
- Training can be interrupted at any time with Ctrl+C
- The implementation includes automatic memory management to prevent OOM errors

## Troubleshooting

1. **Out of Memory Errors**

   - Reduce batch size
   - Reduce maximum sequence length
   - Use gradient accumulation
2. **Dataset Loading Issues**

   - Ensure correct path to dataset
   - Check file permissions
   - Verify CSV format
3. **CUDA Issues**

   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Check GPU memory usage: `nvidia-smi`
