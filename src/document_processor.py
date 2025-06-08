from transformers import TrOCRProcessor, VisionEncoderDecoderModel, T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
import torch
import os
from tqdm import tqdm
import json
from pathlib import Path

class DocumentProcessor:
    def __init__(self, 
                 trocr_model="microsoft/trocr-base-handwritten",
                 t5_model="t5-base",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize TrOCR
        self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        self.trocr_model.to(device)
        self.trocr_model.eval()

        # Initialize T5
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.t5_model.to(device)
        self.t5_model.eval()

        self.device = device

    def preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def extract_text(self, image_path):
        """Extract text from image using TrOCR."""
        # Load and preprocess image
        image = Image.open(image_path)
        image = self.preprocess_image(image)
        pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)
            extracted_text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return extracted_text

    def clean_text(self, text):
        """Clean extracted text using T5."""
        # Prepare input
        input_text = f"fix OCR errors: {text}"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.t5_model.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode
        cleaned_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return cleaned_text

    def process_document(self, image_path):
        """Process a single document image."""
        # Extract text
        extracted_text = self.extract_text(image_path)
        
        # Clean text
        cleaned_text = self.clean_text(extracted_text)
        
        return {
            'image_path': image_path,
            'extracted_text': extracted_text,
            'cleaned_text': cleaned_text
        }

    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory and save results."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).rglob(f'*{ext}'))
        
        # Process each image
        results = []
        for image_path in tqdm(image_files, desc="Processing documents"):
            try:
                result = self.process_document(str(image_path))
                results.append(result)
                
                # Save individual result
                output_file = os.path.join(
                    output_dir, 
                    f"{image_path.stem}_result.json"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Save combined results
        combined_output = os.path.join(output_dir, 'all_results.json')
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process a single image
    image_path = "/root/ocr-repair/data/test/images/5350224/1996/5350224-1996-0001-0037.jpg"
    result = processor.process_document(image_path)
    print("Extracted text:", result['extracted_text'])
    print("Cleaned text:", result['cleaned_text'])
    
    # Process entire directory
    input_dir = "/root/ocr-repair/data/test/images"
    output_dir = "/root/ocr-repair/data/test/results"
    processor.process_directory(input_dir, output_dir) 