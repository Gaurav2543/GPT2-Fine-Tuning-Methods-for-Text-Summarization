import os
import io
import csv
import time
import json
import torch
import psutil
import GPUtil
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm
import torch.nn.init as init
from datetime import datetime
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import get_linear_schedule_with_warmup, GenerationMixin
from transformers import GPT2PreTrainedModel, GPT2Tokenizer, GPT2Model, GPT2Config

# Download dataset from Kaggle and load into /scratch
path = '/scratch/gaurav.bhole/gowrishankarp/newspaper-text-summarization-cnn-dailymail'

def tokenize_data(examples):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for text in examples['text']:
        try:
            if text == 'id,article,highlights':
                continue
                
            # Parse CSV
            reader = csv.reader(io.StringIO(text))
            row = next(reader)
            
            if len(row) >= 3:
                article = row[1]
                highlights = row[2]
                
                # Tokenize article (input)
                input_encoding = tokenizer(
                    article,
                    padding='max_length',
                    truncation=True,
                    max_length=512  # Longer for articles
                )
                
                # Tokenize highlights (labels)
                label_encoding = tokenizer(
                    highlights,
                    padding='max_length',
                    truncation=True,
                    max_length=128  # Shorter for summaries
                )
                
                all_input_ids.append(input_encoding['input_ids'])
                all_attention_masks.append(input_encoding['attention_mask'])
                all_labels.append(label_encoding['input_ids'])
                
        except:
            continue
    
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }

# Modified collate function to handle the new label structure
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])  # Now using actual labels
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'labels': labels
    }

# Dataset loading and processing
dataset = load_dataset('text', data_files={
    'train': path + '/train.csv',
    'validation': path + '/validation.csv',
    'test': path + '/test.csv'
})

# take only 25% of the dataset
dataset['train'] = dataset['train'].select(range(int(len(dataset['train']) * 0.1)))
dataset['validation'] = dataset['validation'].select(range(int(len(dataset['validation']) * 0.1)))
dataset['test'] = dataset['test'].select(range(int(len(dataset['test']) * 0.1)))

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cleanup_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

# # Process the dataset
# tokenized_dataset = dataset.map(
#     tokenize_data,
#     batched=True,
#     batch_size=10000,
#     remove_columns=dataset['train'].column_names,
#     load_from_cache_file=False
# )

# Filter out empty entries
# tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
tokenized_dataset_path = '/scratch/gaurav.bhole/gowrishankarp/newspaper-text-summarization-cnn-dailymail-tokenized0.1'
# tokenized_dataset.save_to_disk(tokenized_dataset_path)

# Load the datasets from the arrow files
def load_arrow_datasets(base_path):
    # Load train, validation and test datasets
    train_dataset = load_from_disk(f"{base_path}/train")
    validation_dataset = load_from_disk(f"{base_path}/validation")
    test_dataset = load_from_disk(f"{base_path}/test")
    
    return train_dataset, validation_dataset, test_dataset

# Load and prepare the data
def prepare_dataloaders(base_path, batch_size=32):
    # Load datasets
    train_dataset, validation_dataset, test_dataset = load_arrow_datasets(base_path)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_dataloader, eval_dataloader, test_dataloader

train_dataloader, eval_dataloader, test_dataloader = prepare_dataloaders(tokenized_dataset_path, batch_size=8)

# print the size of the 3 sets
print(len(train_dataloader), len(eval_dataloader), len(test_dataloader))

def training_summary(training_stats):
    print("\nTraining Summary:")
    print(f"Best Validation Loss: {training_stats['best_validation_loss']:.4f}")
    print(f"Final Test Loss: {training_stats['final_test_loss']:.4f}")
    print(f"Total Epochs Trained: {training_stats['epochs_trained']}")

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels,
                return_dict=True  # Explicitly request dictionary output
            )
            
            total_loss += outputs["loss"].item()
            total_steps += 1

    return total_loss / total_steps

def calculate_rouge_score(model, test_dataloader, tokenizer, device, batch_size=8):
    model.eval()
    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating summaries"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            current_input_ids = input_ids
            current_attention_mask = attention_mask
            max_new_tokens = 100

            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    return_dict=True
                )
                
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), device=device)
                ], dim=-1)
                
                if next_token[0][0].item() == tokenizer.eos_token_id:
                    break
            
            for output in current_input_ids:
                summary = tokenizer.decode(output, skip_special_tokens=True)
                generated_summaries.append(summary)
            
            for labels in batch['labels']:
                reference_summary = tokenizer.decode(labels, skip_special_tokens=True)
                reference_summaries.append(reference_summary)

            # Free up GPU memory
            del input_ids, attention_mask, current_input_ids, current_attention_mask, outputs, next_token_logits, next_token
            torch.cuda.empty_cache()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeLsum': []}

    for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
        score = scorer.score(gen_summary, ref_summary)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    # Average the scores
    avg_scores = {key: sum(value) / len(value) for key, value in scores.items()}

    # Print the average scores
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")

    return avg_scores

class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'rouge1': [],
            'rouge2': [],
            'rougeLsum': [],
            'epochs': [],
            'learning_rate': []
        }
        self.best_scores = {
            'best_eval_loss': float('inf'),
            'best_rouge1': 0,
            'best_rouge2': 0,
            'best_rougeLsum': 0
        }
        self.training_info = {
            'total_params': 0,
            'trainable_params': 0,
            'added_params': 0,
            'training_time': 0,
            'gpu_usage': [],
            'cpu_usage': []
        }
        
        # Create directory for saving metrics
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"metrics_{self.model_name}_{self.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

    def update_train_metrics(self, train_loss, eval_loss, epoch, learning_rate):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['eval_loss'].append(eval_loss)
        self.metrics['epochs'].append(epoch)
        self.metrics['learning_rate'].append(learning_rate)
        
        if eval_loss < self.best_scores['best_eval_loss']:
            self.best_scores['best_eval_loss'] = eval_loss

    def update_rouge_scores(self, val_rouge_scores, test_rouge_scores):
        """Update ROUGE scores for both validation and test sets"""
        self.metrics['rouge1'] = [val_rouge_scores['rouge1'], test_rouge_scores['rouge1']]
        self.metrics['rouge2'] = [val_rouge_scores['rouge2'], test_rouge_scores['rouge2']]
        self.metrics['rougeLsum'] = [val_rouge_scores['rougeLsum'], test_rouge_scores['rougeLsum']]
        
        # Update best scores with validation scores
        for metric in ['rouge1', 'rouge2', 'rougeLsum']:
            self.best_scores[f'best_{metric}'] = val_rouge_scores[metric]


    def update_resource_usage(self, gpu_usage, cpu_usage):
        self.training_info['gpu_usage'].append(gpu_usage)
        self.training_info['cpu_usage'].append(cpu_usage)

    def set_model_info(self, total_params, trainable_params, added_params):
        self.training_info['total_params'] = total_params
        self.training_info['trainable_params'] = trainable_params
        self.training_info['added_params'] = added_params

    def plot_training_curves(self):
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training and Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['epochs'], self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['epochs'], self.metrics['eval_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot 2: Final ROUGE Scores (bar plot)
        plt.subplot(2, 2, 2)
        if len(self.metrics['rouge1']) > 0:  # Only plot if there are ROUGE scores
            metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            val_scores = [self.metrics['rouge1'][0], self.metrics['rouge2'][0], self.metrics['rougeLsum'][0]]
            test_scores = [self.metrics['rouge1'][1], self.metrics['rouge2'][1], self.metrics['rougeLsum'][1]]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, val_scores, width, label='Validation')
            plt.bar(x + width/2, test_scores, width, label='Test')
            plt.title('Final ROUGE Scores')
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True)

        # Plot 3: Resource Usage
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['epochs'], self.training_info['gpu_usage'], label='GPU Usage (%)')
        plt.plot(self.metrics['epochs'], self.training_info['cpu_usage'], label='CPU Usage (%)')
        plt.title('Resource Usage Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True)

        # Plot 4: Learning Rate
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['epochs'], self.metrics['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"lora_training_curves.png")
        plt.close()

    def save_metrics(self):
        # Save all metrics as JSON
        metrics_data = {
            'metrics': self.metrics,
            'best_scores': self.best_scores,
            'training_info': self.training_info
        }
        
        with open(f"lora_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        # Create DataFrame with aligned metrics
        n_epochs = len(self.metrics['epochs'])
        
        # Initialize lists for ROUGE scores with None values
        rouge1_aligned = [None] * n_epochs
        rouge2_aligned = [None] * n_epochs
        rougel_aligned = [None] * n_epochs
        
        # Fill in ROUGE scores at the correct epochs (every 5th epoch)
        # rouge_indices = range(4, n_epochs, 5)  # indices where ROUGE scores were calculated
        # Fill in ROUGE scores at the correct epochs (every epoch)
        rouge_indices = range(n_epochs)
        for i, idx in enumerate(rouge_indices):
            if i < len(self.metrics['rouge1']):
                rouge1_aligned[idx] = self.metrics['rouge1'][i]
                rouge2_aligned[idx] = self.metrics['rouge2'][i]
                rougel_aligned[idx] = self.metrics['rougeLsum'][i]
        
        # Create DataFrame with aligned data
        df = pd.DataFrame({
            'epoch': self.metrics['epochs'],
            'train_loss': self.metrics['train_loss'],
            'eval_loss': self.metrics['eval_loss'],
            'rouge1': rouge1_aligned,
            'rouge2': rouge2_aligned,
            'rougeLsum': rougel_aligned,
            'learning_rate': self.metrics['learning_rate'],
            'gpu_usage': self.training_info['gpu_usage'],
            'cpu_usage': self.training_info['cpu_usage']
        })
        
        # Save to CSV
        df.to_csv(f"lora_metrics.csv", index=False)

    def generate_report(self):
        report = f"""Training Report for {self.model_name}
        
Time: {self.timestamp}

Model Information:
- Total Parameters: {self.training_info['total_params']:,}
- Trainable Parameters: {self.training_info['trainable_params']:,}
- Added Parameters: {self.training_info['added_params']:,}

Best Scores:
- Best Validation Loss: {self.best_scores['best_eval_loss']:.4f}
- Best ROUGE-1: {self.best_scores['best_rouge1']:.4f}
- Best ROUGE-2: {self.best_scores['best_rouge2']:.4f}
- Best ROUGE-L: {self.best_scores['best_rougeLsum']:.4f}

Resource Usage:
- Average GPU Usage: {sum(self.training_info['gpu_usage']) / len(self.training_info['gpu_usage']):.2f}%
- Average CPU Usage: {sum(self.training_info['cpu_usage']) / len(self.training_info['cpu_usage']):.2f}%
"""
        
        with open(f"lora_report.txt", 'w') as f:
            f.write(report)
        
        return report

def print_generated_summaries(summaries):
    print("\nGenerating example summaries from test set...")
    for i, summary in enumerate(summaries, 1):
        print(f"\nExample {i} Summary:")
        print(summary)

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Initialize A with random normal initialization
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0)
        # Initialize B as zero
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Handle different input shapes
        orig_shape = x.shape
        if len(x.shape) == 3:
            # [batch, seq_len, dim] -> [batch * seq_len, dim]
            x = x.view(-1, x.shape[-1])
            
        # Compute LoRA transformation
        result = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        # Restore original shape if needed
        if len(orig_shape) == 3:
            result = result.view(orig_shape[0], orig_shape[1], -1)
            
        return result

class LoRAConv1D(nn.Module):
    def __init__(self, conv_layer, rank=8, alpha=32):
        super().__init__()
        self.conv = conv_layer
        # Freeze the original conv layer
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # Get dimensions from conv layer
        self.in_dim = conv_layer.nf  # input dimension
        self.out_dim = conv_layer.weight.shape[0]  # output dimension
        
        # Add LoRA layer
        self.lora = LoRALayer(self.in_dim, self.out_dim, rank=rank, alpha=alpha)

    def forward(self, x):
        # Original conv output
        orig_output = self.conv(x)
        
        # LoRA output
        lora_output = self.lora(x)
        
        return orig_output + lora_output

class GPT2LoRAModel(GPT2PreTrainedModel):
    def __init__(self, config, rank=8, alpha=32):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
        
        # Enable generation
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers
        self._apply_lora_to_attention_layers(rank, alpha)

        # Tie weights if needed
        self.tie_weights()

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings"""
        self.lm_head.weight = self.transformer.wte.weight

    def _apply_lora_to_attention_layers(self, rank, alpha):
        """Apply LoRA to attention layers"""
        for block in self.transformer.h:
            # The c_attn layer in GPT-2 combines q,k,v projections
            old_c_attn = block.attn.c_attn
            block.attn.c_attn = LoRAConv1D(old_c_attn, rank=rank, alpha=alpha)
            
            # Also apply LoRA to the output projection
            old_c_proj = block.attn.c_proj
            block.attn.c_proj = LoRAConv1D(old_c_proj, rank=rank, alpha=alpha)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if labels is not None:
            labels = labels.to(device)
        if head_mask is not None:
            head_mask = head_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tensors
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels_flat = shift_labels.view(-1)
            
            # Truncate to the shorter sequence
            min_length = min(shift_logits_flat.size(0), shift_labels_flat.size(0))
            shift_logits_flat = shift_logits_flat[:min_length, :]
            shift_labels_flat = shift_labels_flat[:min_length]
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits_flat, shift_labels_flat)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
            "hidden_states": transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
            "attentions": transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load pretrained GPT-2 weights and add LoRA layers"""
        # Extract rank and alpha from kwargs if provided
        rank = kwargs.pop('rank', 8)
        alpha = kwargs.pop('alpha', 32)
        
        # First load the config
        config = kwargs.pop('config', None)
        if config is None:
            config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
            
        # Create model with config
        model = cls(config, rank=rank, alpha=alpha)
        
        # Load pretrained model weights
        model.transformer = GPT2Model.from_pretrained(pretrained_model_name_or_path)
        
        # Initialize the lm_head with the same weights as the token embeddings
        model.lm_head.weight = model.transformer.wte.weight
        
        return model

def train_lora_model(model, train_dataloader, eval_dataloader, test_dataloader, 
                    num_epochs=30, device="cuda", patience=3):
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker("LoRA")
    
    # Calculate and set model parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    added_params = trainable_params  # In LoRA, added params are the trainable params
    metrics_tracker.set_model_info(total_params, trainable_params, added_params)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=len(train_dataloader) * num_epochs
    )

    model.to(device)
    best_eval_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels,
                return_dict=True
            )
            
            loss = outputs["loss"]
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation phase
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_eval_loss = evaluate_model(model, eval_dataloader, device)
        
        # Update metrics
        current_lr = scheduler.get_last_lr()[0]
        metrics_tracker.update_train_metrics(avg_train_loss, avg_eval_loss, epoch + 1, current_lr)
        
        # Update resource usage
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed / GPUtil.getGPUs()[0].memoryTotal * 100
        cpu_usage = psutil.cpu_percent()
        metrics_tracker.update_resource_usage(gpu_usage, cpu_usage)
        
        print(f"Epoch {epoch+1}:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_eval_loss:.4f}")

        # Early stopping check
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "best_lora_model.pt")

    # Load best model for final evaluation
    print("\nTraining completed. Loading best model for final evaluation...")
    model.load_state_dict(best_model_state)
    
    # Calculate ROUGE scores on validation and test sets
    print("Calculating ROUGE scores on validation set...")
    val_rouge_scores = calculate_rouge_score(model, eval_dataloader, tokenizer, device)
    print("Calculating ROUGE scores on test set...")
    test_rouge_scores = calculate_rouge_score(model, test_dataloader, tokenizer, device)
    
    # Update metrics with final ROUGE scores
    metrics_tracker.update_rouge_scores(val_rouge_scores, test_rouge_scores)
    
    # Final test loss
    test_loss = evaluate_model(model, test_dataloader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    training_time = time.time() - start_time
    metrics_tracker.training_info['training_time'] = training_time
    
    # Generate plots and save metrics
    metrics_tracker.plot_training_curves()
    metrics_tracker.save_metrics()
    report = metrics_tracker.generate_report()
    print(report)
    
    return {
        'best_validation_loss': best_eval_loss,
        'final_test_loss': test_loss,
        'epochs_trained': epoch + 1,
        'training_time': training_time,
        'added_params': added_params,
        'gpu_usage': gpu_usage,
        'cpu_usage': cpu_usage,
        'val_rouge_scores': val_rouge_scores,
        'test_rouge_scores': test_rouge_scores
    }

# Initialize model with LoRA
model = GPT2LoRAModel.from_pretrained('gpt2', rank=8, alpha=32)

# Then continue with your training code as before
training_stats = train_lora_model(
    model,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    num_epochs=20,
    device="cuda",
)

print("\nTraining Summary:")
print(f"Best Validation Loss: {training_stats['best_validation_loss']:.4f}")
print(f"Final Test Loss: {training_stats['final_test_loss']:.4f}")
print(f"Total Epochs Trained: {training_stats['epochs_trained']}")
print(f"Training Time: {training_stats['training_time']:.2f} seconds")
print(f"Added Parameters: {training_stats['added_params']}")
print(f"GPU Usage: {training_stats['gpu_usage']:.2f}%")
print(f"CPU Usage: {training_stats['cpu_usage']:.2f}%")

print("\nValidation ROUGE Scores:")
for metric, score in training_stats['val_rouge_scores'].items():
    print(f"{metric}: {score:.4f}")

print("\nTest ROUGE Scores:")
for metric, score in training_stats['test_rouge_scores'].items():
    print(f"{metric}: {score:.4f}")

training_summary(training_stats)
