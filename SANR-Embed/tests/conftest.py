import pytest
import pandas as pd
import numpy as np
import os
import sys
from typing import List

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from models.base import ModelAdapter

class MockModelAdapter(ModelAdapter):
    """
    A mock implementation of ModelAdapter for testing purposes.
    Follows Liskov Substitution Principle - can be used wherever ModelAdapter is expected.
    """
    def __init__(self):
        self.is_trained = False
        self.classify_return_value = "label1"
        self.fine_tune_called = False
        self.reset_called = False

    def translate(self, text: str, target_lang: str) -> str:
        return f"translated_{text}_{target_lang}"

    def embed(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def classify(self, text: str, label_set: List[str]) -> str:
        return self.classify_return_value

    def fine_tune(self, train_texts: List[str], train_labels: List[str], 
                  val_texts: List[str] = None, val_labels: List[str] = None, **kwargs):
        self.fine_tune_called = True
        self.is_trained = True

    def reset(self):
        self.reset_called = True
        self.is_trained = False

    def train(self, df, text_col, label_col):
        # Helper for some tests that expect .train() method (like in main)
        self.fine_tune_called = True
        self.is_trained = True

@pytest.fixture
def mock_adapter():
    return MockModelAdapter()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'text_original': ['text1', 'text2', 'text3'],
        'label_primary': ['A', 'B', 'A'],
        'text_english': ['text1_en', 'text2_en', 'text3_en'],
        'text_chinese': ['text1_zh', 'text2_zh', 'text3_zh']
    })

@pytest.fixture
def temp_data_dir(tmp_path):
    """Creates a temporary data directory with sample csv files."""
    d = tmp_path / "data"
    d.mkdir()
    
    df = pd.DataFrame({
        'text_original': ['text1', 'text2'],
        'label_primary': ['A', 'B']
    })
    df.to_csv(d / "train.csv", index=False)
    df.to_csv(d / "test.csv", index=False)
    return d


