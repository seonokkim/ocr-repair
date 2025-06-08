# Text Restoration Methods Comparison

This project implements and compares three different approaches to text restoration:

1. Masked Language Modeling (MLM)
2. OCR Text Recovery
3. Vision-based Text Restoration
4. Domain Knowledge-based Search

## System Requirements

- Python 3.8+
- Tesseract OCR (with Korean language pack)
    - Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-kor`
- pip packages: see requirements.txt

## Project Structure

```
.
├── data/
│   ├── train/
│   ├── test/
│   │   ├── images/
│   │   ├── labels/
│   │   └── results/
│   ├── domain_knowledge.json
│   └── domain_knowledge_test.json
├── results/           # All test and performance results are saved here
├── src/
│   ├── mlm/
│   │   ├── bert_restoration.py
│   │   └── roberta_restoration.py
│   ├── ocr/
│   │   ├── t5_restoration.py
│   │   └── denoising_autoencoder.py
│   ├── vision/
│   │   ├── trocr_restoration.py
│   │   └── donut_restoration.py
│   ├── document_processor.py
│   ├── create_vector_db.py
│   └── test_methods.py
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
├── notebooks/
│   └── evaluation.ipynb
├── create_domain_knowledge_from_label.py
├── test_performance.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Linux) Install Tesseract and Korean language pack:
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
```

## Usage

### Create Domain Knowledge File

To create a domain knowledge file from label data:
```bash
python create_domain_knowledge_from_label.py
```
This will create a domain knowledge file at `data/domain_knowledge_test.json`.

### Run Performance Test Script

To test all methods on a specific image and save results in the `results/` directory:
```bash
python test_performance.py --image /path/to/image.jpg
```

### Domain Knowledge-based Search

The project includes a domain knowledge-based search approach that uses vector similarity to find and restore text. This method requires a domain knowledge base file (JSON format) containing text samples and their contexts.

Example usage:
```python
from models.search.domain_searcher import DomainSearcher

searcher = DomainSearcher(domain_data_path="data/domain_knowledge.json")
results = searcher.search("text to restore")
```

## Contact

- Email: seonokrkim@gmail.com
- Author: Riley Kim 