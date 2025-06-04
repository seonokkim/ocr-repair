from typing import List, Optional, Union, Dict
import json
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class BertRestorer:
    """Korean BERT-based text restoration model for masked language modeling."""
    
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        device: Optional[str] = None,
        custom_vocab_path: Optional[str] = None
    ):
        """
        Initialize the Korean BERT restorer model.
        
        Args:
            model_name: Name of the pre-trained Korean BERT model to use
            device: Device to run the model on ('cuda' or 'cpu')
            custom_vocab_path: Path to custom vocabulary file for domain-specific terms
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load custom vocabulary if provided
        if custom_vocab_path:
            self._load_custom_vocab(custom_vocab_path)
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_custom_vocab(self, vocab_path: str):
        """Load custom vocabulary for domain-specific terms."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            custom_vocab = json.load(f)
        
        # Add custom tokens to tokenizer
        self.tokenizer.add_tokens(list(custom_vocab.keys()))
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def fine_tune(
        self,
        train_data: List[Dict[str, str]],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on domain-specific data.
        
        Args:
            train_data: List of dictionaries containing 'text' and 'masked_text' pairs
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(train_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['masked_text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save fine-tuned model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def restore(
        self,
        text: str,
        top_k: int = 5,
        return_probs: bool = False
    ) -> Union[str, List[tuple]]:
        """
        Restore masked text using Korean BERT.
        
        Args:
            text: Input text with [MASK] tokens
            top_k: Number of top predictions to return
            return_probs: Whether to return probabilities with predictions
            
        Returns:
            If return_probs is False: Restored text
            If return_probs is True: List of (token, probability) tuples
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Get mask token indices
        mask_token_indices = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        
        # Get top-k predictions for each mask
        results = []
        for idx in mask_token_indices:
            mask_predictions = predictions[0, idx]
            top_k_values, top_k_indices = torch.topk(mask_predictions, top_k)
            
            if return_probs:
                probs = torch.softmax(top_k_values, dim=0)
                tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
                results.extend(list(zip(tokens, probs.tolist())))
            else:
                # Use the most likely prediction
                best_token = self.tokenizer.decode([top_k_indices[0].item()])
                results.append(best_token)
        
        if return_probs:
            return results
        else:
            # Replace masks with predictions
            restored_text = text
            for token in results:
                restored_text = restored_text.replace(self.tokenizer.mask_token, token, 1)
            return restored_text
    
    def batch_restore(
        self,
        texts: List[str],
        top_k: int = 5,
        return_probs: bool = False
    ) -> Union[List[str], List[List[tuple]]]:
        """
        Restore multiple masked texts in batch.
        
        Args:
            texts: List of input texts with [MASK] tokens
            top_k: Number of top predictions to return
            return_probs: Whether to return probabilities with predictions
            
        Returns:
            If return_probs is False: List of restored texts
            If return_probs is True: List of lists of (token, probability) tuples
        """
        return [self.restore(text, top_k, return_probs) for text in texts] 