import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from tqdm import tqdm
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'labels': torch.tensor(label)
        }

class Classifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.n_embd, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        if labels is not None:
            loss = custom_loss(logits, labels)
            return loss, logits
        return logits

def custom_loss(logits, labels):
    soft_labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
    predictions = torch.softmax(logits, dim=1)
    label_range = torch.arange(logits.size(1)).unsqueeze(0).to(labels.device)
    label_distances = torch.abs(label_range - labels.unsqueeze(1))
    weights = 2 ** (1 - label_distances)  # Exponential decay based on distance
    return torch.mean((1 - (predictions * soft_labels) * weights).sum(dim=1))

def train(model, dataloader, optimizer, device='cuda'):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        loss, logits = model(**inputs, labels=labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def predict(model, dataloader, device='cuda'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return predictions

# Setup
model_name = 'Llama-llm-3-8b'
num_labels = 5
epochs = 3
batch_size = 4
learning_rate = 5e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
base_model = LlamaForCausalLM.from_pretrained(model_name)
model = Classifier(base_model, num_labels).to(device)

# Prepare data
df_train = pd.read_csv('path_to_train.csv')  # Adjust path as necessary
df_validate = pd.read_csv('path_to_validate.csv')  # Adjust path as necessary
train_dataset = TextDataset(df_train, tokenizer)
validate_dataset = TextDataset(df_validate, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs
