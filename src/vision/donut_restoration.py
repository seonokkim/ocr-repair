from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json

class DonutRestorer:
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-docvqa"):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.eval()

    def preprocess_image(self, image):
        """Preprocess image for Donut model."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def extract_text(self, image_path):
        """
        Extract text from document image using Donut.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Extracted text
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image = self.preprocess_image(image)
        
        # Prepare task prompt
        task_prompt = "<s_docvqa><s_question>What is the text in this document?</s_question><s_answer>"
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
        
        # Decode
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = sequence.split("<s_answer>")[-1].split("</s_answer>")[0]
        
        return sequence

    def extract_fields(self, image_path, fields=None):
        """
        Extract specific fields from document image.
        
        Args:
            image_path (str): Path to the input image
            fields (List[str]): List of fields to extract
            
        Returns:
            dict: Extracted fields
        """
        if fields is None:
            fields = ["title", "date", "content"]
        
        results = {}
        for field in fields:
            task_prompt = f"<s_docvqa><s_question>What is the {field} in this document?</s_question><s_answer>"
            
            # Load and preprocess image
            image = Image.open(image_path)
            image = self.preprocess_image(image)
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_length=512,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=4,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True
                )
            
            # Decode
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = sequence.split("<s_answer>")[-1].split("</s_answer>")[0]
            
            results[field] = sequence
        
        return results

    def batch_process(self, image_paths, fields=None):
        """
        Process multiple document images.
        
        Args:
            image_paths (List[str]): List of paths to input images
            fields (List[str]): List of fields to extract
            
        Returns:
            List[dict]: List of extracted fields for each image
        """
        results = []
        for image_path in image_paths:
            try:
                if fields:
                    result = self.extract_fields(image_path, fields)
                else:
                    result = {'text': self.extract_text(image_path)}
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({'image_path': image_path, 'error': str(e)})
        
        return results 