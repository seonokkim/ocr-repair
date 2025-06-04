from typing import List, Optional, Union, Dict
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset

class VisionTextRestorer:
    """Korean vision-based text restoration model using TrOCR."""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: Optional[str] = None,
        custom_vocab_path: Optional[str] = None
    ):
        """
        Initialize the Korean vision-based text restorer model.
        
        Args:
            model_name: Name of the pre-trained TrOCR model to use
            device: Device to run the model on ('cuda' or 'cpu')
            custom_vocab_path: Path to custom vocabulary file for domain-specific terms
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Load custom vocabulary if provided
        if custom_vocab_path:
            self._load_custom_vocab(custom_vocab_path)
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_custom_vocab(self, vocab_path: str):
        """Load custom vocabulary for domain-specific terms."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            custom_vocab = json.load(f)
        
        # Add custom tokens to tokenizer
        self.processor.tokenizer.add_tokens(list(custom_vocab.keys()))
        self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
    
    def fine_tune(
        self,
        train_data: List[Dict[str, Union[str, Image.Image]]],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on domain-specific image-text pairs.
        
        Args:
            train_data: List of dictionaries containing 'image' and 'text' pairs
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(train_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Process images
            pixel_values = self.processor(
                examples['image'],
                return_tensors="pt"
            ).pixel_values
            
            # Process text
            labels = self.processor.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
            
            return {
                'pixel_values': pixel_values,
                'labels': labels['input_ids']
            }
        
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
        self.processor.save_pretrained(output_dir)
    
    def restore_from_image(
        self,
        image_path: Union[str, Path],
        max_length: int = 128,
        num_beams: int = 4,
        return_probs: bool = False
    ) -> Union[str, tuple]:
        """
        Restore Korean text from an image.
        
        Args:
            image_path: Path to the input image
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            return_probs: Whether to return generation probabilities
            
        Returns:
            If return_probs is False: Restored text
            If return_probs is True: Tuple of (restored text, probability)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=return_probs
            )
        
        # Decode output
        restored_text = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0]
        
        if return_probs:
            # Calculate sequence probability
            probs = torch.exp(outputs.sequences_scores[0])
            return restored_text, probs.item()
        
        return restored_text
    
    def batch_restore(
        self,
        image_paths: List[Union[str, Path]],
        max_length: int = 128,
        num_beams: int = 4,
        return_probs: bool = False
    ) -> Union[List[str], List[tuple]]:
        """
        Restore Korean text from multiple images in batch.
        
        Args:
            image_paths: List of paths to input images
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            return_probs: Whether to return generation probabilities
            
        Returns:
            If return_probs is False: List of restored texts
            If return_probs is True: List of (restored text, probability) tuples
        """
        return [
            self.restore_from_image(path, max_length, num_beams, return_probs)
            for path in image_paths
        ]
    
    def preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple = (384, 384),
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input PIL Image
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        if normalize:
            pixel_values = pixel_values / 255.0
        
        return pixel_values.to(self.device) 