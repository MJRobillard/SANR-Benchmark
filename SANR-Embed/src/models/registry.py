from typing import Dict, Type, List, Any, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from .base import ModelAdapter
from .deepseek_v3.adapter import DeepSeekV3Adapter
from .llama3.adapter import Llama3Adapter
from .qwen25.adapter import Qwen25Adapter

# Registry to hold model instances
MODEL_REGISTRY: Dict[str, ModelAdapter] = {
    "deepseek-v3": DeepSeekV3Adapter(),
    "llama-3": Llama3Adapter(),
    "qwen-2.5": Qwen25Adapter(),
}

def register_model(name: str, adapter: ModelAdapter):
    """
    Register a model adapter instance with a given name.
    """
    MODEL_REGISTRY[name] = adapter

def get_model(name: str) -> ModelAdapter:
    """
    Retrieve a model adapter by name.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

def run_kfold_finetuning(model_name: str, texts: List[str], labels: List[str], k: int = 5, label_set: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """
    Run K-Fold Cross Validation fine-tuning for a registered model.
    
    Args:
        model_name: Name of the model in registry
        texts: List of texts
        labels: List of ground truth labels
        k: Number of folds
        label_set: List of all possible labels (optional, inferred if None)
        **kwargs: Arguments passed to fine_tune
        
    Returns:
        Dictionary with average F1 and per-fold scores.
    """
    model = get_model(model_name)
    
    X = np.array(texts)
    y = np.array(labels)
    
    if label_set is None:
        label_set = sorted(list(set(labels)))
    
    # Robust Splitting Logic
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()
    
    if min_samples < k:
        print(f"Warning: Minority class has {min_samples} samples, less than k={k}. Falling back to non-stratified KFold.")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
    f1_scores = []
    
    fold = 1
    for train_index, val_index in kf.split(X, y):
        print(f"Starting Fold {fold}/{k}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Reset model to base state
        model.reset()
        
        # Fine-tune on training split
        # Note: We pass lists as adapters expect lists
        model.fine_tune(
            train_texts=X_train.tolist(), 
            train_labels=y_train.tolist(), 
            val_texts=X_val.tolist(), 
            val_labels=y_val.tolist(), 
            **kwargs
        )
        
        # Evaluate on validation split
        preds = []
        for text in X_val:
            preds.append(model.classify(text, label_set))
            
        score = f1_score(y_val, preds, average='macro', zero_division=0)
        f1_scores.append(score)
        print(f"Fold {fold} Macro F1: {score:.4f}")
        fold += 1
        
    avg_f1 = float(np.mean(f1_scores))
    print(f"K-Fold Complete. Average Macro F1: {avg_f1:.4f}")
    
    return {
        "f1_macro_avg": avg_f1,
        "f1_scores": f1_scores,
        "k": k
    }

