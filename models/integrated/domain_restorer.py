from typing import List, Dict, Optional, Union
from pathlib import Path

from models.mlm.bert_restorer import BertRestorer
from models.search.domain_searcher import DomainSearcher

class DomainRestorer:
    """Integrated domain-specific text restoration combining fine-tuning and search."""
    
    def __init__(
        self,
        domain_data_path: str,
        model_name: str = "klue/bert-base",
        device: Optional[str] = None,
        search_threshold: float = 0.7,
        use_fine_tuned: bool = True
    ):
        """
        Initialize the domain restorer.
        
        Args:
            domain_data_path: Path to domain-specific knowledge base
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
            search_threshold: Similarity threshold for search-based restoration
            use_fine_tuned: Whether to use fine-tuned model if available
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.search_threshold = search_threshold
        self.use_fine_tuned = use_fine_tuned
        
        # Initialize components
        self.searcher = DomainSearcher(
            domain_data_path=domain_data_path,
            model_name=model_name,
            device=device
        )
        
        self.bert_restorer = BertRestorer(
            model_name=model_name,
            device=device,
            custom_vocab_path=domain_data_path
        )
    
    def restore(
        self,
        text: str,
        top_k: int = 5,
        use_search_first: bool = True
    ) -> Union[str, List[Dict[str, Union[str, float]]]]:
        """
        Restore text using both search and fine-tuned model.
        
        Args:
            text: Input text to restore
            top_k: Number of top results to return
            use_search_first: Whether to try search-based restoration first
            
        Returns:
            Restored text or list of potential matches
        """
        if use_search_first:
            # Try search-based restoration first
            search_results = self.searcher.restore(
                text,
                top_k=top_k,
                threshold=self.search_threshold
            )
            
            # If good match found, return it
            if isinstance(search_results, str):
                return search_results
            
            # If no good match, try fine-tuned model
            if self.use_fine_tuned:
                return self.bert_restorer.restore(text)
            
            # Return search results if no fine-tuned model
            return search_results
        else:
            # Try fine-tuned model first
            if self.use_fine_tuned:
                restored = self.bert_restorer.restore(text)
                
                # Verify result with search
                search_results = self.searcher.search(
                    restored,
                    top_k=1,
                    threshold=self.search_threshold
                )
                
                if search_results:
                    return restored
                
            # Fall back to search if fine-tuned model fails or not available
            return self.searcher.restore(
                text,
                top_k=top_k,
                threshold=self.search_threshold
            )
    
    def fine_tune(
        self,
        train_data: List[Dict[str, str]],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on domain-specific data.
        
        Args:
            train_data: List of dictionaries containing 'text' and 'masked_text' pairs
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        self.bert_restorer.fine_tune(
            train_data=train_data,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
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
        self.searcher.add_to_knowledge_base(text, context)
    
    def save_knowledge_base(self, path: str):
        """Save domain knowledge base to file."""
        self.searcher.save_knowledge_base(path) 