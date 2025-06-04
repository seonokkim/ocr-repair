from pathlib import Path

from models.mlm.bert_restorer import BertRestorer
from models.ocr.ocr_corrector import OCRCorrector
from models.vision.vision_restorer import VisionTextRestorer

def demo_mlm_restoration():
    """Demonstrate MLM-based text restoration."""
    print("\n=== MLM-based Text Restoration ===")
    restorer = BertRestorer()
    
    # Example texts with masks
    texts = [
        "The cat sat on the [MASK]",
        "I love to [MASK] in the morning",
        "The [MASK] is the capital of France"
    ]
    
    print("\nInput texts:")
    for text in texts:
        print(f"- {text}")
    
    print("\nRestored texts:")
    restored = restorer.batch_restore(texts)
    for original, restored in zip(texts, restored):
        print(f"- {original} -> {restored}")

def demo_ocr_correction():
    """Demonstrate OCR text correction."""
    print("\n=== OCR Text Correction ===")
    corrector = OCRCorrector()
    
    # Example noisy OCR texts
    texts = [
        "T1e c@t sa7 on t!e mat",
        "H3llo w0rld!",
        "Th3 qu1ck br0wn f0x"
    ]
    
    print("\nInput texts:")
    for text in texts:
        print(f"- {text}")
    
    print("\nCorrected texts:")
    corrected = corrector.batch_correct(texts)
    for original, corrected in zip(texts, corrected):
        print(f"- {original} -> {corrected}")

def demo_vision_restoration():
    """Demonstrate vision-based text restoration."""
    print("\n=== Vision-based Text Restoration ===")
    restorer = VisionTextRestorer()
    
    # Example image paths (replace with actual image paths)
    image_paths = [
        "path/to/damaged_document1.jpg",
        "path/to/damaged_document2.jpg"
    ]
    
    print("\nProcessing images:")
    for path in image_paths:
        if Path(path).exists():
            restored = restorer.restore_from_image(path)
            print(f"- {path} -> {restored}")
        else:
            print(f"- {path} (file not found)")

def main():
    """Run all demos."""
    print("Text Restoration AI Demo")
    print("=======================")
    
    # Run MLM demo
    demo_mlm_restoration()
    
    # Run OCR correction demo
    demo_ocr_correction()
    
    # Run vision restoration demo
    demo_vision_restoration()

if __name__ == "__main__":
    main() 