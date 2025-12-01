from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class ModelAdapter(ABC):
    """
    Base interface that all models must implement for the SANR-Embed benchmark.
    """

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text using the model's own translation capabilities.
        
        Args:
            text: The source text to translate
            target_lang: The target language code (e.g., 'en', 'zh')
            
        Returns:
            The translated text
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for any given text.
        
        Args:
            text: The input text
            
        Returns:
            A numpy array representing the embedding vector
        """
        pass

    @abstractmethod
    def classify(self, text: str, label_set: List[str]) -> str:
        """
        Return a predicted label for classification tasks.
        
        Args:
            text: The input text to classify
            label_set: A list of valid labels/classes
            
        Returns:
            The predicted label (must be one of label_set)
        """
        pass

    def fine_tune(self, train_texts: List[str], train_labels: List[str], 
                  val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None, 
                  **kwargs):
        """
        Optional fine-tuning stage.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts (optional)
            val_labels: List of validation labels (optional)
            **kwargs: Additional arguments (e.g., epochs, learning_rate)
        """
        pass

    def reset(self):
        """
        Reset the model to its initial state (untrained or pre-trained base).
        Useful for cross-validation.
        """
        pass


    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        Optional optimization for batch embeddings.
        Default implementation loops over embed().
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Matrix of embeddings (n_samples, embedding_dim)
        """
        return np.array([self.embed(t) for t in texts])

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """
        Optional optimization for batch translation.
        Default implementation loops over translate().
        """
        return [self.translate(t, target_lang) for t in texts]

