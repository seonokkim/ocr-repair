from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from tqdm import tqdm
import pickle

class VectorDB:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_documents(self, documents):
        """Add documents to the vector database."""
        # Generate embeddings
        texts = [doc['cleaned_text'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents
        self.documents.extend(documents)

    def search(self, query, k=5):
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'document': self.documents[idx],
                    'score': float(distances[0][i])
                })
        
        return results

    def save(self, directory):
        """Save the vector database."""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save documents
        with open(os.path.join(directory, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, directory):
        """Load a saved vector database."""
        # Load FAISS index
        index = faiss.read_index(os.path.join(directory, 'index.faiss'))
        
        # Load documents
        with open(os.path.join(directory, 'documents.pkl'), 'rb') as f:
            documents = pickle.load(f)
        
        # Create instance
        instance = cls()
        instance.index = index
        instance.documents = documents
        return instance

def create_vector_db(input_dir, output_dir):
    """Create vector database from processed documents."""
    # Initialize vector database
    vector_db = VectorDB()
    
    # Load all results
    results_file = os.path.join(input_dir, 'all_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Add documents to vector database
    print("Adding documents to vector database...")
    vector_db.add_documents(documents)
    
    # Save vector database
    print("Saving vector database...")
    vector_db.save(output_dir)
    
    return vector_db

if __name__ == "__main__":
    # Create vector database from processed documents
    input_dir = "/root/ocr-repair/data/test/results"
    output_dir = "/root/ocr-repair/data/test/vector_db"
    
    vector_db = create_vector_db(input_dir, output_dir)
    
    # Example search
    query = "What is the main topic of the document?"
    results = vector_db.search(query, k=3)
    
    print("\nSearch results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']}")
        print(f"Text: {result['document']['cleaned_text'][:200]}...")
        print(f"Image: {result['document']['image_path']}") 