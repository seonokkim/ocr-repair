# Text Restoration Methods Comparison

This project implements and compares three different approaches to text restoration:

1. Masked Language Modeling (MLM)
2. OCR Text Recovery
3. Vision-based Text Restoration

## Project Structure

```
.
├── data/
│   ├── train/
│   ├── test/
│   │   ├── images/
│   │   ├── labels/
│   │   └── results/
│   └── validation/
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
├── notebooks/
│   └── evaluation.ipynb
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

3. Download required models:
```bash
python src/download_models.py
```

## Usage

### 1. MLM-based Restoration
```python
from src.mlm.roberta_restoration import RoBERTaRestorer

restorer = RoBERTaRestorer()
restored_text = restorer.restore("The cat sat on the [MASK]")
```

### 2. OCR Text Recovery
```python
from src.ocr.t5_restoration import T5Restorer

restorer = T5Restorer()
restored_text = restorer.restore("T1e c@t sa7 on t!e mat")
```

### 3. Vision-based Restoration
```python
from src.vision.trocr_restoration import TrOCRRestorer

restorer = TrOCRRestorer()
restored_text = restorer.restore("path/to/image.jpg")
```

### 4. Combined Processing
```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document("path/to/image.jpg")
```

### 5. Vector Database Creation
```python
from src.create_vector_db import create_vector_db

vector_db = create_vector_db(
    input_dir="data/test/results",
    output_dir="data/test/vector_db"
)
```

## Testing Methods

To test all methods on a specific image:
```bash
python src/test_methods.py
```

This will generate three types of result files in `data/test/results/`:
1. `results.json`: Detailed results in JSON format
2. `results.txt`: Results in plain text format
3. `results.md`: Results in markdown format

## Models Used

### 1. MLM Models
- BERT (bert-base-uncased)
- RoBERTa (roberta-base)

### 2. OCR Recovery Models
- T5 (t5-base)
- Denoising Autoencoder (custom implementation)

### 3. Vision Models
- TrOCR (microsoft/trocr-base-handwritten)
- Donut (naver-clova-ix/donut-base-finetuned-docvqa)

## Evaluation Metrics

The following metrics are used for evaluation:
- BLEU Score
- ROUGE Score
- BERTScore
- Character Error Rate (CER)
- Word Error Rate (WER)

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