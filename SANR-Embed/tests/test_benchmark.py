import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import json

# Add root to path to import main
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import main
from models import registry

def test_load_data_success(temp_data_dir):
    train, test = main.load_data(str(temp_data_dir))
    assert len(train) == 2
    assert len(test) == 2
    assert 'text_original' in train.columns

def test_load_data_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        main.load_data(str(tmp_path))

def test_evaluate_model(mock_adapter, sample_df):
    mock_adapter.classify_return_value = 'A'
    
    metrics = main.evaluate_model(
        model=mock_adapter,
        df=sample_df,
        text_col='text_original',
        label_col='label_primary'
    )
    
    assert metrics['f1_macro'] >= 0.0
    assert 'report' in metrics
    # Since all predictions are 'A', and ground truth has 'A' and 'B'.
    # Recall for A will be 1.0, for B 0.0.

def test_ensure_translations_cache_hit(mock_adapter, sample_df, tmp_path):
    cache_file = tmp_path / "cached_trans.csv"
    sample_df.to_csv(cache_file, index=False)
    
    # Should read from file and not call translate_batch
    with patch.object(mock_adapter, 'translate_batch') as mock_trans:
        df = main.ensure_translations(mock_adapter, sample_df, 'en', str(cache_file))
        mock_trans.assert_not_called()
        assert len(df) == 3

def test_ensure_translations_perform(mock_adapter, sample_df, tmp_path):
    cache_file = tmp_path / "new_trans.csv"
    
    with patch.object(mock_adapter, 'translate_batch', return_value=['t1', 't2', 't3']) as mock_trans:
        df = main.ensure_translations(mock_adapter, sample_df, 'en', str(cache_file))
        
        mock_trans.assert_called_once()
        assert 'text_english' in df.columns
        assert df['text_english'].tolist() == ['t1', 't2', 't3']
        assert os.path.exists(cache_file)

def test_run_task_a(mock_adapter, temp_data_dir, tmp_path):
    """
    Integration test for Task A using a mock model.
    """
    output_dir = tmp_path / "results"
    train_df, test_df = main.load_data(str(temp_data_dir))
    
    # Patch get_model to return our mock
    with patch('main.get_model', return_value=mock_adapter):
        f1 = main.run_task_a("mock_model", train_df, test_df, str(output_dir))
        
        assert isinstance(f1, float)
        assert (output_dir / "task_a_results.json").exists()
        assert mock_adapter.fine_tune_called  # verify training happened

def test_run_task_a_fine_tune(mock_adapter, temp_data_dir, tmp_path):
    """
    Test explicit fine-tuning flag.
    """
    output_dir = tmp_path / "results_ft"
    train_df, test_df = main.load_data(str(temp_data_dir))
    
    mock_adapter.fine_tune_called = False # Reset
    
    with patch('main.get_model', return_value=mock_adapter):
        main.run_task_a("mock_model", train_df, test_df, str(output_dir), fine_tune=True)
        
        assert mock_adapter.fine_tune_called

def test_run_task_a_kfold(mock_adapter, temp_data_dir, tmp_path):
    """
    Test K-Fold mode.
    """
    output_dir = tmp_path / "results_kfold"
    train_df, test_df = main.load_data(str(temp_data_dir))
    
    # We need to patch run_kfold_finetuning since it's imported in main
    with patch('main.get_model', return_value=mock_adapter):
        with patch('main.run_kfold_finetuning', return_value={'f1_macro_avg': 0.8}) as mock_kfold:
            
            f1 = main.run_task_a("mock_model", train_df, test_df, str(output_dir), k_folds=5)
            
            assert f1 == 0.8
            mock_kfold.assert_called_once()
            assert (output_dir / "task_a_kfold_results.json").exists()

def test_run_task_b(mock_adapter, temp_data_dir, tmp_path):
    """
    Integration test for Task B using a mock model.
    """
    output_dir = tmp_path / "results"
    train_df, test_df = main.load_data(str(temp_data_dir))
    
    # Mock translate_batch to return dummy translations matching length
    def side_effect_translate(texts, lang):
        return [f"trans_{lang}" for _ in texts]
    
    mock_adapter.translate_batch = side_effect_translate
    
    with patch('main.get_model', return_value=mock_adapter):
        main.run_task_b("mock_model", train_df, test_df, str(output_dir))
        
        assert (output_dir / "task_b_results.json").exists()
        trans_dir = output_dir / "translations"
        assert trans_dir.exists()
        # Should verify that 'es', 'en', 'zh' keys are in results json
        with open(output_dir / "task_b_results.json") as f:
            res = json.load(f)
            assert 'f1_es' in res
            assert 'f1_en' in res
            assert 'f1_zh' in res
