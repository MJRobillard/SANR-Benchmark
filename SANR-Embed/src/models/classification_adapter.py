import os
import torch
import pytorch_lightning as pl
import numpy as np
from typing import List, Optional
from transformers import BertModel, BertTokenizer
from torch import nn
from .base import ModelAdapter
from processing.translate_dataset import translate_text


class SANRClassifier(pl.LightningModule):
    def __init__(self, num_classes=33, model_name='dccuchile/bert-base-spanish-wwm-cased'):
        super().__init__()
        self.save_hyperparameters()
        # Initialize base model. 
        # We use from_pretrained to get the correct configuration (layers, hidden size, etc.)
        # The weights will be overwritten by the checkpoint load.
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use pooler output (index 1) which corresponds to [CLS] token embedding processed by a linear layer + tanh
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)


class ClassificationAdapter(ModelAdapter):
    # Fixed label set (33 classes)
    # 'year' is moved to index 7 (effectively 'Date') to align with model weights
    LABELS = [
        'Agreement', 'Amount', 'Asset', 'Binding', 'Circumstance', 
        'Copete', 'Corroboratio', 'year', 'Delegation', 'Disallowance', 
        'Dispositio', 'Expositio', 'Header', 'Institutio', 'Intitulatio', 
        'Invocatio', 'Legal', 'Limitation', 'Name', 'Notificatio', 
        'Oath', 'Parties', 'Penalty', 'Place', 'Property', 
        'Reassurance', 'Residence', 'Scope', 'Signatures', 'Slave', 
        'Type', 'Validation', 'Value'
    ]

    def __init__(self, model_base_path: str):
        """
        Args:
            model_base_path: Path to 'SANR-Embed/src/models/Classification/Classification' 
                             (containing 'classifier_checkpoint.ckpt' and 'classifier_tokenizer')
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_base_path = model_base_path
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is not None:
            return

        tokenizer_path = os.path.join(self.model_base_path, "classifier_tokenizer")
        ckpt_base_dir = os.path.join(self.model_base_path, "classifier_checkpoint.ckpt", "lightning_logs")
        
        # Find the checkpoint file
        ckpt_file = None
        if os.path.exists(ckpt_base_dir):
            for root, dirs, files in os.walk(ckpt_base_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        ckpt_file = os.path.join(root, file)
                        # Prefer later versions or epochs if multiple? 
                        # For now, take the first deep one we find or specific one if known.
                        # Based on user files: QTag-epoch=119-val_loss=0.07.ckpt in version_1
                        if "epoch=119" in file: 
                            break 
                if ckpt_file and "epoch=119" in ckpt_file: break
            
            # Fallback if specific epoch not found
            if not ckpt_file:
                for root, dirs, files in os.walk(ckpt_base_dir):
                     for file in files:
                        if file.endswith(".ckpt"):
                            ckpt_file = os.path.join(root, file)
                            break
                     if ckpt_file: break

        if not ckpt_file:
            raise ValueError(f"No .ckpt file found in {ckpt_base_dir}")

        print(f"Loading Classifier from {ckpt_file}...")
        # Load model
        # We catch strict loading errors in case there are minor mismatches, but generally expect match
        try:
            self.model = SANRClassifier.load_from_checkpoint(ckpt_file)
        except Exception as e:
            print(f"Warning: Strict loading failed ({e}), trying with strict=False...")
            self.model = SANRClassifier.load_from_checkpoint(ckpt_file, strict=False)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading Tokenizer from {tokenizer_path}...")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("Classifier loaded.")

    def classify(self, text: str, label_set: List[str]) -> str:
        self._load_model()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        # Map index to label using the fixed label list corresponding to model training
        if pred_idx < len(self.LABELS):
             return self.LABELS[pred_idx]
        return "Unknown"

    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            # Use pooler output
            embedding = outputs[1]
            
        return embedding[0].cpu().numpy()

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

    def fine_tune(self, train_texts: List[str], train_labels: List[str], 
                  val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None, 
                  **kwargs):
        self._load_model()
        
        # Map labels
        label_map = {label: i for i, label in enumerate(self.LABELS)}
        
        def encode_data(texts, labels):
            # Check labels validity
            valid_labels = [l for l in labels if l in label_map]
            valid_texts = [t for t, l in zip(texts, labels) if l in label_map]
            if len(valid_texts) < len(texts):
                print(f"Warning: {len(texts) - len(valid_texts)} samples dropped due to invalid labels.")
            
            if not valid_texts:
                raise ValueError("No valid samples to train on.")

            encodings = self.tokenizer(valid_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
            # Move to cpu first for dataset creation
            label_ids = torch.tensor([label_map[l] for l in valid_labels])
            return torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], label_ids)

        train_dataset = encode_data(train_texts, train_labels)
        val_dataset = encode_data(val_texts, val_labels) if val_texts and val_labels else None
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=kwargs.get('batch_size', 8), shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=kwargs.get('batch_size', 8)) if val_dataset else None
        
        trainer = pl.Trainer(
            max_epochs=kwargs.get('epochs', 3),
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            logger=False
        )
        
        trainer.fit(self.model, train_loader, val_loader)

    def reset(self):
        self.model = None
        self.tokenizer = None


