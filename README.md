# Text Restoration Methods Comparison

This project implements and compares three different approaches to text restoration:

1. Masked Language Modeling (MLM)
2. OCR Text Recovery
3. Vision-based Text Restoration

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
│   └── validation/
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
├── notebooks/
│   └── evaluation.ipynb
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

### Run Performance Test Script

To test all methods on a specific image and save results in the `results/` directory:
```bash
python test_performance.py --image /path/to/image.jpg
```

- 결과는 `results/` 폴더에 다음과 같이 저장됩니다:
    - 각 방법별 추출 텍스트: `mlm_[이미지명]_[타임스탬프].txt`, `ocr_denoising_[이미지명]_[타임스탬프].txt`, `vision_based_[이미지명]_[타임스탬프].txt`
    - 전체 결과 요약: `performance_report_[타임스탬프].json|txt|csv`
    - 개별 결과 상세: `performance_test_[이미지명]_[타임스탬프].json`

#### 예시
```bash
python test_performance.py --image data/test/images/5350224/1996/5350224-1996-0001-0037.jpg
```

### 옵션
- `--limit N` : 최대 N개의 샘플만 평가
- `--data-dir DIR` : 테스트 이미지가 있는 디렉토리 지정

## 결과 파일 구조
- `results/`
    - `mlm_*.txt`, `ocr_denoising_*.txt`, `vision_based_*.txt` : 각 방법별 추출 텍스트
    - `performance_test_*.json` : 개별 이미지별 상세 결과
    - `performance_report_*.json|txt|csv` : 전체 성능 요약

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

Riley Kim (seonokrkim@gmail.com) 