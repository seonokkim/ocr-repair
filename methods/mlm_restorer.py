from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class MLMRestorer:
    def __init__(self):
        self.model_name = "klue/bert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        
    def restore_text(self, text: str) -> dict:
        """Restore text using MLM approach"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Get the most likely tokens
        predicted_tokens = torch.argmax(predictions, dim=-1)
        
        # Convert back to text
        restored_text = self.tokenizer.decode(predicted_tokens[0])
        
        return {
            "original_text": text,
            "restored_text": restored_text,
            "confidence_score": float(torch.max(torch.softmax(predictions[0], dim=-1)).item())
        } 