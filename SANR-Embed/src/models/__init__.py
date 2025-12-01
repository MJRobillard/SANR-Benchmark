import os
from .base import ModelAdapter
from .registry import MODEL_REGISTRY, register_model, get_model
from .baseline_adapter import LogisticRegressionAdapter
from .masked_lm_adapter import MaskedLMAdapter
from .classification_adapter import ClassificationAdapter

# Register default models
register_model("logreg", LogisticRegressionAdapter())

# Register Masked Language Model
# Path is relative to this file: ./masked_language_model
mlm_path = os.path.join(os.path.dirname(__file__), 'masked_language_model')
# Only register if the directory exists to avoid errors on systems without the model
if os.path.exists(mlm_path) and os.path.isdir(mlm_path):
    register_model("masked_language_model", MaskedLMAdapter(mlm_path))

# Register Classification Model
# Path is relative to this file: ./Classification/Classification
cls_path = os.path.join(os.path.dirname(__file__), 'Classification', 'Classification')
if os.path.exists(cls_path) and os.path.isdir(cls_path):
    register_model("classification_model", ClassificationAdapter(cls_path))

__all__ = ['ModelAdapter', 'MODEL_REGISTRY', 'register_model', 'get_model', 'LogisticRegressionAdapter', 'MaskedLMAdapter', 'ClassificationAdapter']
