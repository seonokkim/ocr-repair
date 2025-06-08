import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

class DenoisingAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        self.decoder = nn.LSTM(
            hidden_dim * 2,  # bidirectional
            hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Encode
        encoded, _ = self.encoder(embedded)
        
        # Decode
        decoded, _ = self.decoder(encoded)
        
        # Output
        output = self.output(decoded)
        return output

class OCRDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class DenoisingAutoencoderTrainer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())

    def train(self, train_data, val_data=None, batch_size=32, epochs=10):
        train_dataset = OCRDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            val_dataset = OCRDataset(val_data, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
            
            if val_data:
                val_loss = self.evaluate(val_loader)
                print(f'Validation Loss: {val_loss:.4f}')

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def restore(self, text):
        """
        Restore noisy OCR text using the trained model.
        
        Args:
            text (str): Noisy OCR text
            
        Returns:
            str: Restored text
        """
        self.model.eval()
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(encoding['input_ids'])
            predictions = outputs.argmax(dim=-1)
            
        restored_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        return restored_text

def create_noisy_data(text, noise_level=0.1):
    """
    Create synthetic noisy OCR data.
    
    Args:
        text (str): Original text
        noise_level (float): Probability of character corruption
        
    Returns:
        str: Noisy text
    """
    chars = list(text)
    for i in range(len(chars)):
        if np.random.random() < noise_level:
            # Replace with similar looking character
            if chars[i].isalpha():
                chars[i] = np.random.choice('abcdefghijklmnopqrstuvwxyz')
            elif chars[i].isdigit():
                chars[i] = np.random.choice('0123456789')
    
    return ''.join(chars) 