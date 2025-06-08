from transformers import (
    BertForMaskedLM, BertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    TrOCRProcessor, VisionEncoderDecoderModel
)
import os

def download_models():
    """Download all required pre-trained models."""
    print("Downloading BERT model...")
    BertForMaskedLM.from_pretrained("bert-base-uncased")
    BertTokenizer.from_pretrained("bert-base-uncased")

    print("Downloading T5 model...")
    T5ForConditionalGeneration.from_pretrained("t5-base")
    T5Tokenizer.from_pretrained("t5-base")

    print("Downloading TrOCR model...")
    TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    print("All models downloaded successfully!")

if __name__ == "__main__":
    # Create cache directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    download_models() 