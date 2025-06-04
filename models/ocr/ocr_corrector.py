from typing import List, Optional, Union, Dict
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class OCRCorrector:
    """Korean OCR text correction model using T5."""
    
    def __init__(
        self,
        model_name: str = "KETI-AIR/kor-t5-base",
        device: Optional[str] = None,
        custom_vocab_path: Optional[str] = None
    ):
        """
        Initialize the Korean OCR corrector model.
        
        Args:
            model_name: Name of the pre-trained Korean T5 model to use
            device: Device to run the model on ('cuda' or 'cpu')
            custom_vocab_path: Path to custom vocabulary file for domain-specific terms
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load custom vocabulary if provided
        if custom_vocab_path:
            self._load_custom_vocab(custom_vocab_path)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        Fine-tune the model on domain-specific OCR data.
        
        Args:
            train_data: List of dictionaries containing 'noisy_text' and 'clean_text' pairs
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(train_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['noisy_text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['clean_text'],
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
            inputs['labels'] = labels['input_ids']
            return inputs
        
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
    
    def correct(
        self,
        text: str,
        max_length: int = 128,
        num_beams: int = 4,
        return_probs: bool = False
    ) -> Union[str, tuple]:
        """
        Correct noisy Korean OCR text.
        
        Args:
            text: Input noisy OCR text
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            return_probs: Whether to return generation probabilities
            
        Returns:
            If return_probs is False: Corrected text
            If return_probs is True: Tuple of (corrected text, probability)
        """
        # Prepare input
        inputs = self.tokenizer(
            f"한글 OCR 오류 수정: {text}",
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate correction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=return_probs
            )
        
        # Decode output
        corrected_text = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        if return_probs:
            # Calculate sequence probability
            probs = torch.exp(outputs.sequences_scores[0])
            return corrected_text, probs.item()
        
        return corrected_text
    
    def batch_correct(
        self,
        texts: List[str],
        max_length: int = 128,
        num_beams: int = 4,
        return_probs: bool = False
    ) -> Union[List[str], List[tuple]]:
        """
        Correct multiple noisy Korean OCR texts in batch.
        
        Args:
            texts: List of input noisy OCR texts
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            return_probs: Whether to return generation probabilities
            
        Returns:
            If return_probs is False: List of corrected texts
            If return_probs is True: List of (corrected text, probability) tuples
        """
        return [
            self.correct(text, max_length, num_beams, return_probs)
            for text in texts
        ]
    
    def add_noise(
        self,
        text: str,
        noise_level: float = 0.1,
        noise_types: List[str] = None
    ) -> str:
        """
        Add synthetic Korean OCR noise to text for training.
        
        Args:
            text: Input clean text
            noise_level: Probability of adding noise to each character
            noise_types: Types of noise to add (e.g., ['substitution', 'deletion', 'insertion'])
            
        Returns:
            Text with synthetic OCR noise
        """
        if noise_types is None:
            noise_types = ['substitution', 'deletion', 'insertion']
        
        # Implementation of Korean-specific noise addition logic
        # This is a placeholder - implement actual noise generation logic
        return text  # TODO: Implement Korean noise generation 