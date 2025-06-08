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
│   └── validation/
├── src/
│   ├── mlm/
│   │   ├── bert_restoration.py
│   │   └── roberta_restoration.py
│   ├── ocr/
│   │   ├── seq2seq_restoration.py
│   │   └── t5_restoration.py
│   └── vision/
│       ├── trocr_restoration.py
│       └── donut_restoration.py
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
from src.mlm.bert_restoration import BERTRestorer

restorer = BERTRestorer()
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

## Evaluation

Run the evaluation notebook to compare the performance of different methods:
```bash
jupyter notebook notebooks/evaluation.ipynb
```

## Models Used

- MLM: BERT-base, RoBERTa-base
- OCR Recovery: T5-base, BART-base
- Vision: TrOCR, Donut

## Metrics

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