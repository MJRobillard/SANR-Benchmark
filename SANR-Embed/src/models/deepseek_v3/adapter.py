from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base import ModelAdapter
import torch
import yaml
import os
import numpy as np

class DeepSeekV3Adapter(ModelAdapter):
    def __init__(self):
        cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.path = os.path.expanduser("~/.cache/sanr_models/deepseek_v3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return

        if not os.path.exists(self.path) or not os.listdir(self.path):
             raise ValueError(f"Model weights not found at {self.path}. Please run 'python scripts/download_all_models.py'")
             
        print(f"Loading DeepSeek-V3 from {self.path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # Use float16 for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def translate(self, text, target_lang):
        self._load()
        prompt = self.config["translation_prompt"].replace("{{lang}}", target_lang)
        full_prompt = f"{prompt}\n\n{text}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.tokenizer.pad_token_id)
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Basic prompt stripping
        if decoded.startswith(full_prompt):
            return decoded[len(full_prompt):].strip()
        return decoded

    def embed(self, text):
        self._load()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config["context_length"]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            
        # Mean pooling
        mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        return embedding.cpu().numpy()[0]

    def classify(self, text, label_set):
        self._load()
        prompt = f"Text: {text}\nLabels: {', '.join(label_set)}\nBest label:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20, pad_token_id=self.tokenizer.pad_token_id)
            
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = result[len(prompt):].strip()
        return gen_text.split('\n')[0].strip()




