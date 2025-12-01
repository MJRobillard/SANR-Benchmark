from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from .base import ModelAdapter
from processing.translate_dataset import translate_text
import os

class MaskedLMAdapter(ModelAdapter):
    def __init__(self, model_base_path: str):
        """
        Args:
            model_base_path: Path to the directory containing 'mlm_model' and 'mlm_tokenizer' directories.
        """
        self.model_base_path = model_base_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.model is not None:
            return
            
        tokenizer_path = os.path.join(self.model_base_path, "mlm_tokenizer")
        model_path = os.path.join(self.model_base_path, "mlm_model")
        
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        if not os.path.exists(model_path):
             raise ValueError(f"Model not found at {model_path}")

        print(f"Loading Masked LM from {self.model_base_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Masked LM loaded.")

    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings[0].cpu().numpy()

    def classify(self, text: str, label_set: List[str]) -> str:
        # Masked LM is not a classifier
        raise NotImplementedError("MaskedLMAdapter does not support classification directly.")

    def translate(self, text: str, target_lang: str) -> str:
        # Fallback to NLLB
        lang_map = {
            'en': 'eng_Latn',
            'zh': 'zho_Hans',
            'es': 'spa_Latn'
        }
        nllb_tgt = lang_map.get(target_lang, target_lang)
        
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

