# OCR Repair

A comprehensive text restoration system that combines multiple approaches for handling masked text, noisy OCR text, and vision-based text restoration.

## Features

- **Masked Text Restoration**: Uses BERT-based models for restoring masked text
- **OCR Text Correction**: Implements T5-based models for correcting noisy OCR text
- **Vision-based Text Restoration**: Utilizes TrOCR for restoring text from images
- **Domain-Specific Fine-tuning**: Supports fine-tuning models for specific domains
- **Search-Based Restoration**: Implements vector similarity search for domain-specific text restoration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/seonokkim/ocr-repair.git
cd ocr-repair
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
```bash
python scripts/download_models.py
```

## Usage

### Basic Usage

```python
from models.integrated.domain_restorer import DomainRestorer

# Initialize the domain restorer
restorer = DomainRestorer(
    domain_data_path="data/domain_knowledge.json",
    model_name="klue/bert-base"
)

# Restore text
restored_text = restorer.restore("masked text to restore")
```

### Fine-tuning

```python
# Prepare training data
train_data = [
    {"text": "original text", "masked_text": "masked text"},
    # ... more training examples
]

# Fine-tune the model
restorer.fine_tune(
    train_data=train_data,
    output_dir="models/fine_tuned",
    num_epochs=3
)
```

### Adding Domain Knowledge

```python
# Add new text to domain knowledge base
restorer.add_to_knowledge_base(
    "new domain text",
    context={"type": "document", "category": "legal"}
)
```

## Project Structure

```
ocr-repair/
├── data/
│   └── domain_knowledge.json
├── models/
│   ├── mlm/
│   │   └── bert_restorer.py
│   ├── ocr/
│   │   └── ocr_corrector.py
│   ├── vision/
│   │   └── vision_restorer.py
│   ├── search/
│   │   └── domain_searcher.py
│   └── integrated/
│       └── domain_restorer.py
├── scripts/
│   └── download_models.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Riley Kim - seonokrkim@gmail.com

Project Link: [https://github.com/seonokkim/ocr-repair](https://github.com/seonokkim/ocr-repair) 