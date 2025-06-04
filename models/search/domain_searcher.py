from typing import List, Dict, Optional, Union
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from transformers import AutoTokenizer

class DomainSearcher:
    """Search-based domain text restoration using vector similarity."""
    
    def __init__(
        self,
        domain_data_path: str,
        model_name: str = "klue/bert-base",
        device: Optional[str] = None,
        index_type: str = "faiss"  # or "tfidf"
    ):
        """
        Initialize the domain searcher.
        
        Args:
            domain_data_path: Path to domain-specific knowledge base
            model_name: Name of the pre-trained model to use for embeddings
            device: Device to run the model on ('cuda' or 'cpu')
            index_type: Type of search index to use ('faiss' or 'tfidf')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_type = index_type
        
        # Load domain data
        self.domain_data = self._load_domain_data(domain_data_path)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Build search index
        self._build_index()
    
    def _load_domain_data(self, data_path: str) -> Dict:
        """Load domain-specific knowledge base."""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_index(self):
        """Build search index from domain data."""
        if self.index_type == "tfidf":
            # TF-IDF based indexing
            self.vectorizer = TfidfVectorizer()
            self.index = self.vectorizer.fit_transform(self.domain_data.keys())
        else:
            # FAISS based indexing
            # Convert texts to embeddings
            texts = list(self.domain_data.keys())
            embeddings = self._get_embeddings(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts using the tokenizer."""
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.tokenizer.model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Search for similar texts in domain knowledge base.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to return
            threshold: Similarity threshold for results
            
        Returns:
            List of dictionaries containing matched text and similarity score
        """
        if self.index_type == "tfidf":
            # TF-IDF based search
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.index).flatten()
        else:
            # FAISS based search
            query_embedding = self._get_embeddings([query])
            similarities, indices = self.index.search(
                query_embedding.astype('float32'),
                top_k
            )
            similarities = similarities[0]
            indices = indices[0]
        
        # Get results above threshold
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                if self.index_type == "tfidf":
                    text = list(self.domain_data.keys())[i]
                else:
                    text = list(self.domain_data.keys())[indices[i]]
                
                results.append({
                    "text": text,
                    "score": float(score),
                    "context": self.domain_data[text]
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def restore(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Union[str, List[Dict[str, Union[str, float]]]]:
        """
        Restore text using domain knowledge.
        
        Args:
            text: Input text to restore
            top_k: Number of top results to return
            threshold: Similarity threshold for results
            
        Returns:
            If threshold is met: Restored text
            If threshold is not met: List of potential matches
        """
        results = self.search(text, top_k, threshold)
        
        if results and results[0]["score"] >= threshold:
            return results[0]["text"]
        else:
            return results
    
    def add_to_knowledge_base(
        self,
        text: str,
        context: Optional[Dict] = None
    ):
        """
        Add new text to domain knowledge base.
        
        Args:
            text: Text to add
            context: Additional context information
        """
        self.domain_data[text] = context or {}
        self._build_index()  # Rebuild index with new data
    
    def save_knowledge_base(self, path: str):
        """Save domain knowledge base to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.domain_data, f, ensure_ascii=False, indent=2) 