from transformers import BertForMaskedLM, BertTokenizer
import torch

class BERTRestorer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def restore(self, text):
        """
        Restore masked text using BERT MLM.
        
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