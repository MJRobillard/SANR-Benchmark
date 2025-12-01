import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from models import registry

def test_register_and_get_model(mock_adapter):
    """
    Test that models can be registered and retrieved.
    """
    # Clear registry for isolation
    with patch.dict(registry.MODEL_REGISTRY, {}, clear=True):
        registry.register_model("mock_model", mock_adapter)
        
        retrieved = registry.get_model("mock_model")
        assert retrieved == mock_adapter
        
        with pytest.raises(ValueError):
            registry.get_model("non_existent_model")

def test_run_kfold_finetuning(mock_adapter):
    """
    Test the k-fold fine-tuning logic.
    Uses Dependency Injection via the registry.
    """
    texts = ["text1", "text2", "text3", "text4"]
    labels = ["A", "B", "A", "B"]
    
    with patch.dict(registry.MODEL_REGISTRY, {}, clear=True):
        registry.register_model("mock_model", mock_adapter)
        
        # Mock KFold to be deterministic and simple
        # The original code uses StratifiedKFold or KFold. 
        # We trust sklearn's implementation, so we focus on ensuring our adapter hooks are called.
        
        # We'll patch KFold in the registry module if we wanted to control splits strictly,
        # but for this test, we just want to ensure fine_tune is called k times.
        
        # Since the implementation does a loop, we can wrap the mock_adapter.fine_tune to count calls.
        # But mock_adapter.fine_tune is already instrumented in our MockModelAdapter (sets fine_tune_called=True).
        # To count, let's wrap it in a Mock.
        
        mock_adapter.fine_tune = MagicMock()
        mock_adapter.reset = MagicMock()
        
        results = registry.run_kfold_finetuning(
            model_name="mock_model",
            texts=texts,
            labels=labels,
            k=2
        )
        
        # Assertions
        assert results['k'] == 2
        assert len(results['f1_scores']) == 2
        assert "f1_macro_avg" in results
        
        # Check interactions
        assert mock_adapter.reset.call_count == 2
        assert mock_adapter.fine_tune.call_count == 2

def test_kfold_fallback(mock_adapter, capsys):
    """
    Test that it falls back to KFold when minority class samples < k.
    """
    # One sample for class 'C', but k=2. Should trigger warning.
    texts = ["text1", "text2", "text3"]
    labels = ["A", "A", "C"] 
    
    with patch.dict(registry.MODEL_REGISTRY, {}, clear=True):
        registry.register_model("mock_model", mock_adapter)
        
        registry.run_kfold_finetuning(
            model_name="mock_model",
            texts=texts,
            labels=labels,
            k=2
        )
        
        captured = capsys.readouterr()
        assert "Falling back to non-stratified KFold" in captured.out






