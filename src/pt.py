import os
import io
import csv
import json
import time
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
from transformers import GPT2PreTrainedModel, GPT2Tokenizer, GPT2Model
from transformers import get_linear_schedule_with_warmup, GenerationMixin

# Download dataset from Kaggle and load into /scratch
path = '/scratch/gaurav.bhole/gowrishankarp/newspaper-text-summarization-cnn-dailymail'

# Load dataset
dataset = load_dataset('text', data_files={
    'train': path + '/train.csv',
    'validation': path + '/validation.csv',
    'test': path + '/test.csv'
})

# Tokenize the dataset
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

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])  # Now using actual labels
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'labels': labels
    }

# take only 0.25% of the dataset (as per your code)
dataset['train'] = dataset['train'].select(range(int(len(dataset['train']) * 0.1)))
dataset['validation'] = dataset['validation'].select(range(int(len(dataset['validation']) * 0.1)))
dataset['test'] = dataset['test'].select(range(int(len(dataset['test']) * 0.1)))

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cleanup_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

# Process the dataset
tokenized_dataset = dataset.map(
    tokenize_data,
    batched=True,
    batch_size=10000,
    remove_columns=dataset['train'].column_names,
    load_from_cache_file=False
)

# Filter out empty entries
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
tokenized_dataset_path = '/scratch/gaurav.bhole/gowrishankarp/newspaper-text-summarization-cnn-dailymail-tokenized0.1'
tokenized_dataset.save_to_disk(tokenized_dataset_path)

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
            labels = batch['labels'].to(device)  # Actual reference summaries

            # Generate summaries
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
            
            # Decode generated summaries
            for output in current_input_ids:
                summary = tokenizer.decode(output, skip_special_tokens=True)
                generated_summaries.append(summary)
            
            # Decode reference summaries from labels
            for label in labels:
                reference_summary = tokenizer.decode(label, skip_special_tokens=True)
                reference_summaries.append(reference_summary)

            # Free up GPU memory
            del input_ids, attention_mask, current_input_ids, current_attention_mask, outputs
            torch.cuda.empty_cache()

    # Calculate ROUGE scores
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
            'learning_rate': [],
            'gpu_usage': [],
            'cpu_usage': []
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
            'training_time': 0
        }
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("training_metrics", exist_ok=True)

    def set_model_info(self, total_params, trainable_params, added_params):
        """Set model parameters information"""
        self.training_info['total_params'] = total_params
        self.training_info['trainable_params'] = trainable_params
        self.training_info['added_params'] = added_params

    def update_train_metrics(self, train_loss, eval_loss, epoch, learning_rate, gpu_usage, cpu_usage):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['eval_loss'].append(eval_loss)
        self.metrics['epochs'].append(epoch)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['gpu_usage'].append(gpu_usage)
        self.metrics['cpu_usage'].append(cpu_usage)
        
        if eval_loss < self.best_scores['best_eval_loss']:
            self.best_scores['best_eval_loss'] = eval_loss

    def update_rouge_scores(self, val_rouge_scores, test_rouge_scores):
        self.metrics['rouge1'] = [val_rouge_scores['rouge1'], test_rouge_scores['rouge1']]
        self.metrics['rouge2'] = [val_rouge_scores['rouge2'], test_rouge_scores['rouge2']]
        self.metrics['rougeLsum'] = [val_rouge_scores['rougeLsum'], test_rouge_scores['rougeLsum']]
        
        # Update best scores with validation scores
        for metric in ['rouge1', 'rouge2', 'rougeLsum']:
            self.best_scores[f'best_{metric}'] = val_rouge_scores[metric]

    def save_metrics(self):
        # Save metrics as CSV
        df = pd.DataFrame({
            'epoch': self.metrics['epochs'],
            'train_loss': self.metrics['train_loss'],
            'eval_loss': self.metrics['eval_loss'],
            'learning_rate': self.metrics['learning_rate'],
            'gpu_usage': self.metrics['gpu_usage'],
            'cpu_usage': self.metrics['cpu_usage']
        })
        
        # Add final ROUGE scores to a separate DataFrame
        rouge_df = pd.DataFrame({
            'metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
            'validation_score': [self.metrics['rouge1'][0], self.metrics['rouge2'][0], self.metrics['rougeLsum'][0]],
            'test_score': [self.metrics['rouge1'][1], self.metrics['rouge2'][1], self.metrics['rougeLsum'][1]]
        })
        
        # Save to CSV
        df.to_csv("training_metrics/pt_training_metrics.csv", index=False)
        rouge_df.to_csv("training_metrics/pt_rouge_scores.csv", index=False)
        
        # Save all data as JSON
        metrics_data = {
            'metrics': self.metrics,
            'best_scores': self.best_scores,
            'training_info': self.training_info
        }
        
        with open("training_metrics/pt_all_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=4)

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
        plt.plot(self.metrics['epochs'], self.metrics['gpu_usage'], label='GPU Usage (%)')
        plt.plot(self.metrics['epochs'], self.metrics['cpu_usage'], label='CPU Usage (%)')
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
        plt.savefig("training_metrics/pt_training_curves.png")
        plt.close()

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
- Average GPU Usage: {sum(self.metrics['gpu_usage']) / len(self.metrics['gpu_usage']):.2f}%
- Average CPU Usage: {sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']):.2f}%
"""
        
        with open("training_metrics/pt_report.txt", 'w') as f:
            f.write(report)
        
        return report

def print_generated_summaries(summaries):
    print("\nGenerating example summaries from test set...")
    for i, summary in enumerate(summaries, 1):
        print(f"\nExample {i} Summary:")
        print(summary)

class SoftPromptEmbedding(nn.Module):
    def __init__(self, wte: nn.Embedding, n_tokens: int = 20, random_range: float = 0.5):
        super().__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.Parameter(torch.zeros(n_tokens, wte.weight.size(1)))

        with torch.no_grad():
            for i in range(n_tokens):
                self.learned_embedding[i] = self.wte.weight[i].clone() + \
                    torch.zeros_like(self.wte.weight[i]).uniform_(-random_range, random_range)

        # Initialize learned embedding
        init.normal_(self.learned_embedding)

    def forward(self, tokens):
        input_embedding = self.wte(tokens)
        batch_size = tokens.shape[0]
        learned_embedding = self.learned_embedding.repeat(batch_size, 1, 1)
        return torch.cat([learned_embedding, input_embedding], dim=1)

class GPT2PromptTuningModel(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.soft_prompt = SoftPromptEmbedding(self.transformer.wte)
        self.soft_prompt.learned_embedding.requires_grad = True

        # Enable generation
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

        # Freeze transformer and LM head parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

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
        if inputs_embeds is None:
            if past_key_values is None:
                # First step: include soft prompt
                inputs_embeds = self.soft_prompt(input_ids)
                if attention_mask is not None:
                    prompt_attention_mask = torch.ones(
                        attention_mask.shape[0], 
                        self.soft_prompt.n_tokens, 
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                # Subsequent steps: normal embedding
                inputs_embeds = self.transformer.wte(input_ids)

        # Forward through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Create new labels tensor padded to match input length
            if past_key_values is None:
                # Account for the soft prompt tokens
                target_length = lm_logits.size(1) - self.soft_prompt.n_tokens
                padded_labels = torch.full(
                    (labels.shape[0], target_length),
                    -100,  # padding token id for loss calculation
                    device=labels.device,
                    dtype=labels.dtype
                )
                # Copy the actual labels into the padded tensor
                copy_length = min(target_length, labels.size(1))
                padded_labels[:, :copy_length] = labels[:, :copy_length]
                
                # Shift logits and labels
                shift_logits = lm_logits[..., self.soft_prompt.n_tokens:-1, :].contiguous()
                shift_labels = padded_labels[..., 1:].contiguous()
            else:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            # Reshape tensors to match expected shape for cross entropy
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

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
    
def train_prompt_tuning_model(model, train_dataloader, eval_dataloader, test_dataloader, num_epochs=30, device="cuda"):
    optimizer = torch.optim.Adam(model.soft_prompt.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * num_epochs)

    model.to(device)
    best_eval_loss = float('inf')
    best_model_state = None
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    added_params = trainable_params

    metrics_tracker = MetricsTracker("PromptTuning")
    metrics_tracker.set_model_info(total_params, trainable_params, added_params)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Added parameters: {added_params}")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # # Add debug prints for the first batch of first epoch
            # if epoch == 0 and batch_idx == 0:
            #     print(f"\nDebug - Input shapes:")
            #     print(f"input_ids shape: {input_ids.shape}")
            #     print(f"attention_mask shape: {attention_mask.shape}")
            #     print(f"labels shape: {labels.shape}")
                
            #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            #     # print(f"logits shape: {outputs['logits'].shape}")
            #     # print(f"loss value: {outputs['loss'].item()}\n")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)   
            
            loss = outputs["loss"]
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_eval_loss = evaluate_model(model, eval_dataloader, device)

        current_lr = scheduler.get_last_lr()[0]
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed / GPUtil.getGPUs()[0].memoryTotal * 100 if GPUtil.getGPUs() else 0
        cpu_usage = psutil.cpu_percent()

        metrics_tracker.update_train_metrics(
            train_loss=avg_train_loss,
            eval_loss=avg_eval_loss,
            epoch=epoch + 1,
            learning_rate=current_lr,
            gpu_usage=gpu_usage,
            cpu_usage=cpu_usage
        )

        print(f"Epoch {epoch+1}:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_eval_loss:.4f}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "best_prompt_tuning_model.pt")

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
    print(metrics_tracker.generate_report())

    return {
        'best_validation_loss': best_eval_loss,
        'final_test_loss': test_loss,
        'epochs_trained': num_epochs,
        'training_time': training_time,
        'added_params': added_params,
        'gpu_usage': gpu_usage,
        'cpu_usage': cpu_usage,
        'val_rouge_scores': val_rouge_scores,
        'test_rouge_scores': test_rouge_scores
    }

# Initialize model
model = GPT2PromptTuningModel.from_pretrained('gpt2')

# Train model
training_stats = train_prompt_tuning_model(
    model, 
    train_dataloader, 
    eval_dataloader, 
    test_dataloader,
    num_epochs=20,
    device="cuda"
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

