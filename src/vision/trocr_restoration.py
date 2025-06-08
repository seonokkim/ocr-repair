from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class TrOCRRestorer:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.eval()

    def restore(self, image_path):
        """
        Restore text from an image using TrOCR.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Restored text
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return generated_text

    def batch_restore(self, image_paths):
        """
        Restore text from multiple images.
        
        Args:
            image_paths (List[str]): List of paths to input images
            
        Returns:
            List[str]: List of restored texts
        """
        return [self.restore(path) for path in image_paths]

    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Add image preprocessing steps here
        # e.g., contrast enhancement, noise removal, etc.
        return image 