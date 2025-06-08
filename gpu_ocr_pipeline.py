import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.vision.trocr_restoration import TrOCRRestorer
from src.ocr.t5_restoration import T5Restorer
from sentence_transformers import SentenceTransformer

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize models on GPU
trocr = TrOCRRestorer()
trocr.model = trocr.model.to(device)
t5 = T5Restorer()
t5.model = t5.model.to(device)
encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Directories
image_root = "data/test/images"
embedding_dir = "data/embeddings"
results_dir = "results"
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Recursively gather images
image_paths = glob.glob(os.path.join(image_root, "**", "*.jpg"), recursive=True)
results = []

# Batch size for embedding (adjust as needed)
batch_size = 16

# OCR and clean text
ocr_texts = []
cleaned_texts = []
valid_image_paths = []

for image_path in tqdm(image_paths, desc="OCR and Cleaning"):
    try:
        ocr_text = trocr.restore(image_path)
    except Exception as e:
        print(f"TrOCR failed for {image_path}: {e}")
        ocr_text = ""
    try:
        cleaned_text = t5.restore(ocr_text) if ocr_text else ""
    except Exception as e:
        print(f"T5 failed for {image_path}: {e}")
        cleaned_text = ocr_text
    ocr_texts.append(ocr_text)
    cleaned_texts.append(cleaned_text)
    valid_image_paths.append(image_path)

# Embedding in batches
all_embeddings = []
for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Embedding"):
    batch_texts = cleaned_texts[i:i+batch_size]
    try:
        batch_embeddings = encoder.encode(batch_texts, convert_to_numpy=True)
    except Exception as e:
        print(f"Embedding failed for batch {i}: {e}")
        batch_embeddings = np.zeros((len(batch_texts), 384))
    all_embeddings.append(batch_embeddings)
all_embeddings = np.vstack(all_embeddings)

# Save embeddings and results
for image_path, ocr_text, cleaned_text, embedding in zip(valid_image_paths, ocr_texts, cleaned_texts, all_embeddings):
    # Save embedding with subfolder structure
    rel_path = os.path.relpath(image_path, image_root)
    emb_path = os.path.join(embedding_dir, rel_path + ".npy")
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.save(emb_path, embedding)
    results.append({
        "image_path": image_path,
        "ocr_text": ocr_text,
        "cleaned_text": cleaned_text,
        "embedding_path": emb_path
    })

df = pd.DataFrame(results)
df.to_csv(os.path.join(results_dir, "ocr_pipeline_results.csv"), index=False)
print("Pipeline complete. Results saved to results/ocr_pipeline_results.csv") 