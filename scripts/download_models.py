import os
from pathlib import Path
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

def download_models():
    """Download all required pre-trained Korean models."""
    # Create models directory if it doesn't exist
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to download
    model_configs = {
        "klue/bert-base": {
            "type": "mlm",
            "model_class": AutoModelForMaskedLM,
            "tokenizer_class": AutoTokenizer
        },
        "KETI-AIR/kor-t5-base": {
            "type": "seq2seq",
            "model_class": AutoModelForSeq2SeqLM,
            "tokenizer_class": AutoTokenizer
        },
        "microsoft/trocr-base-handwritten": {
            "type": "vision",
            "model_class": VisionEncoderDecoderModel,
            "tokenizer_class": TrOCRProcessor
        }
    }
    
    print("Downloading pre-trained Korean models...")
    
    for model_name, config in model_configs.items():
        print(f"\nDownloading {model_name}...")
        
        # Create model-specific directory
        model_dir = models_dir / model_name.replace("/", "_")
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Download model
            model = config["model_class"].from_pretrained(model_name)
            model.save_pretrained(model_dir)
            
            # Download tokenizer
            tokenizer = config["tokenizer_class"].from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            
            print(f"✓ Successfully downloaded {model_name}")
            
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {str(e)}")
    
    print("\nAll Korean models downloaded successfully!")

if __name__ == "__main__":
    download_models() 