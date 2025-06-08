from transformers import RobertaForMaskedLM, RobertaTokenizer
import torch

class RoBERTaRestorer:
    def __init__(self, model_name="roberta-base"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def restore(self, text):
        """
        Restore masked text using RoBERTa MLM.
        
        Args:
            text (str): Input text with [MASK] token
            
        Returns:
            str: Restored text
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            
        # Replace [MASK] with predicted token
        restored_tokens = self.tokenizer.convert_ids_to_tokens(predictions[0])
        restored_text = self.tokenizer.convert_tokens_to_string(restored_tokens)
        
        return restored_text

    def batch_restore(self, texts):
        """
        Restore multiple masked texts.
        
        Args:
            texts (List[str]): List of input texts with [MASK] tokens
            
        Returns:
            List[str]: List of restored texts
        """
        return [self.restore(text) for text in texts] 