import pytest
from PIL import Image
import torch

from models.mlm.bert_restorer import BertRestorer
from models.ocr.ocr_corrector import OCRCorrector
from models.vision.vision_restorer import VisionTextRestorer

@pytest.fixture
def bert_restorer():
    return BertRestorer()

@pytest.fixture
def ocr_corrector():
    return OCRCorrector()

@pytest.fixture
def vision_restorer():
    return VisionTextRestorer()

def test_bert_restorer(bert_restorer):
    """Test BERT-based text restoration."""
    # Test single text restoration
    text = "The cat sat on the [MASK]"
    restored = bert_restorer.restore(text)
    assert isinstance(restored, str)
    assert restored != text
    assert "[MASK]" not in restored
    
    # Test batch restoration
    texts = ["The [MASK] is blue", "I love [MASK]"]
    restored_batch = bert_restorer.batch_restore(texts)
    assert isinstance(restored_batch, list)
    assert len(restored_batch) == len(texts)
    assert all(isinstance(t, str) for t in restored_batch)

def test_ocr_corrector(ocr_corrector):
    """Test OCR text correction."""
    # Test single text correction
    text = "T1e c@t sa7 on t!e mat"
    corrected = ocr_corrector.correct(text)
    assert isinstance(corrected, str)
    assert corrected != text
    
    # Test batch correction
    texts = ["H3llo w0rld!", "Th3 qu1ck br0wn f0x"]
    corrected_batch = ocr_corrector.batch_correct(texts)
    assert isinstance(corrected_batch, list)
    assert len(corrected_batch) == len(texts)
    assert all(isinstance(t, str) for t in corrected_batch)
    
    # Test noise addition
    clean_text = "The quick brown fox"
    noisy_text = ocr_corrector.add_noise(clean_text, noise_level=0.1)
    assert isinstance(noisy_text, str)

def test_vision_restorer(vision_restorer):
    """Test vision-based text restoration."""
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (100, 100), color='white')
    
    # Test image preprocessing
    preprocessed = vision_restorer.preprocess_image(dummy_image)
    assert isinstance(preprocessed, torch.Tensor)
    assert preprocessed.shape[0] == 1  # Batch size of 1
    
    # Test with probability return
    text, prob = vision_restorer.restore_from_image(
        dummy_image,
        return_probs=True
    )
    assert isinstance(text, str)
    assert isinstance(prob, float)
    assert 0 <= prob <= 1

def test_model_initialization():
    """Test model initialization with different devices."""
    # Test CPU initialization
    bert_cpu = BertRestorer(device='cpu')
    assert bert_cpu.device == 'cpu'
    
    # Test GPU initialization if available
    if torch.cuda.is_available():
        bert_gpu = BertRestorer(device='cuda')
        assert bert_gpu.device == 'cuda'
    
    # Test automatic device selection
    bert_auto = BertRestorer()
    assert bert_auto.device in ['cpu', 'cuda'] 