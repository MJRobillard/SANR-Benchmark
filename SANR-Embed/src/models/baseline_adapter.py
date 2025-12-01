from typing import List, Optional
import numpy as np
import pandas as pd
from .base import ModelAdapter
from classifiers.baselines import BaselineClassifier
from processing.translate_dataset import translate_text

class LogisticRegressionAdapter(ModelAdapter):
    def __init__(self):
        self.clf = BaselineClassifier()
        self.is_trained = False

    def translate(self, text: str, target_lang: str) -> str:
        """
        Logistic Regression cannot translate. 
        We use NLLB-200 as the standard fallback for baselines.
        """
        # Map target_lang to NLLB codes if needed, or rely on translate_text defaults/logic
        # translate_text expects list
        # We'll assume target_lang is like 'eng_Latn' or 'en'
        # The benchmark uses 'facebook/nllb-200-distilled-600M'
        
        # Simple mapping for the benchmark languages
        lang_map = {
            'en': 'eng_Latn',
            'zh': 'zho_Hans',
            'es': 'spa_Latn'
        }
        nllb_tgt = lang_map.get(target_lang, target_lang)
        
        # Assuming source is Spanish (17th century, but NLLB spa_Latn is closest)
        results = translate_text(
            [text], 
            model_name="facebook/nllb-200-distilled-600M", 
            src_lang="spa_Latn", 
            tgt_lang=nllb_tgt,
            batch_size=1,
            mock=False
        )
        return results[0] if results else ""

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """
        Batch translation using NLLB (efficient).
        """
        lang_map = {
            'en': 'eng_Latn',
            'zh': 'zho_Hans',
            'es': 'spa_Latn'
        }
        nllb_tgt = lang_map.get(target_lang, target_lang)
        
        return translate_text(
            texts, 
            model_name="facebook/nllb-200-distilled-600M", 
            src_lang="spa_Latn", 
            tgt_lang=nllb_tgt,
            batch_size=8,
            mock=False
        )

    def embed(self, text: str) -> np.ndarray:
        if not self.is_trained:
            return np.array([])
        
        # Extract TF-IDF vector
        tfidf = self.clf.pipeline.named_steps['tfidf']
        # Transform returns sparse matrix
        vector = tfidf.transform([text]).toarray()[0]
        return vector

    def classify(self, text: str, label_set: List[str]) -> str:
        if not self.is_trained:
             raise RuntimeError("Model must be trained before classification.")
        
        return self.clf.pipeline.predict([text])[0]

    def train(self, df, text_col, label_col):
        """Extra method for trainable baselines"""
        self.clf.train(df, text_col=text_col, label_col=label_col)
        self.is_trained = True

    def fine_tune(self, train_texts: List[str], train_labels: List[str], 
                  val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None, 
                  **kwargs):
        """
        Fine-tune (train) the logistic regression model.
        """
        df = pd.DataFrame({'text': train_texts, 'label': train_labels})
        self.clf.train(df, text_col='text', label_col='label')
        self.is_trained = True

    def reset(self):
        self.clf = BaselineClassifier()
        self.is_trained = False


