import argparse
import pandas as pd
import os
import sys
import json
import warnings
from sklearn.metrics import classification_report, f1_score

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models import get_model, ModelAdapter
from models.registry import run_kfold_finetuning

warnings.filterwarnings("ignore")

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train/Test files not found in {data_dir}. Please run src/processing/create_splits.py")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def evaluate_model(model: ModelAdapter, df: pd.DataFrame, text_col: str, label_col: str):
    """
    Generic evaluation loop for any ModelAdapter.
    """
    print(f"Evaluating on {len(df)} samples using column '{text_col}'...")
    
    # Filter out NaN labels and texts
    df = df[df[label_col].notna()]
    df = df[df[text_col].notna()]
    
    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].tolist()
    
    if not texts:
        return {'f1_macro': 0.0, 'report': {}}

    # Get label set from ground truth
    label_set = sorted(list(set(y_true)))
    
    preds = []
    # Simple loop for prediction
    for text in texts:
        try:
            p = model.classify(text, label_set)
            preds.append(p)
        except Exception as e:
            print(f"Error classifying sample: {e}")
            preds.append("error")
            
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    
    return {
        'f1_macro': f1,
        'report': report
    }

def run_task_a(model_name, train_df, test_df, output_dir, fine_tune=False, k_folds=0):
    print("\n" + "="*40)
    print(f"TASK A: Native Legal Classification ({model_name})")
    print("="*40)
    
    model = get_model(model_name)
    
    # Case 1: K-Fold Cross Validation
    if k_folds > 0:
        print(f"Running {k_folds}-Fold Cross Validation (Fine-tuning Comparison)...")
        
        # Combine train + test for a full evaluation or just use train_df?
        # Standard practice: CV on Train set to tune, or CV on all available data if data is scarce.
        # Given the "Comparison of fine tuning" request, likely we want robust metrics.
        # Let's use the Train DF for CV to simulate development performance.
        
        texts = train_df['text_original'].astype(str).tolist()
        labels = train_df['label_primary'].tolist()
        
        results = run_kfold_finetuning(model_name, texts, labels, k=k_folds)
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "task_a_kfold_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results['f1_macro_avg']

    # Case 2: Standard Train/Test (Fine-tuning explicitly requested or Baseline)
    # Baselines (like LogisticRegression) MUST train.
    # We check if fine_tune is requested OR if the model has a legacy 'train' method that implies it's a baseline.
    
    should_train = fine_tune or hasattr(model, 'train')
    
    if should_train:
        print(f"Fine-tuning/Training {model_name} on Spanish data...")
        
        # Prefer fine_tune interface
        # Convert columns to lists
        train_texts = train_df['text_original'].astype(str).tolist()
        train_labels = train_df['label_primary'].tolist()
        
        # Some models might need validation data during fine-tuning
        val_texts = test_df['text_original'].astype(str).tolist()
        val_labels = test_df['label_primary'].tolist()

        model.fine_tune(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)
    else:
        print(f"Zero-shot Evaluation (No fine-tuning requested)...")
        
    metrics = evaluate_model(model, test_df, 'text_original', 'label_primary')
    
    print(f"\nResult - Macro F1: {metrics['f1_macro']:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "task_a_results.json"), "w") as f:
        json.dump(metrics['report'], f, indent=2)
        
    return metrics['f1_macro']

def ensure_translations(model, df, target_lang, cache_path):
    """
    Translate dataframe using model capabilities or cache.
    """
    if os.path.exists(cache_path):
        print(f"Loading translations from {cache_path}")
        return pd.read_csv(cache_path)
        
    print(f"Translating {len(df)} records to {target_lang}...")
    texts = df['text_original'].fillna("").tolist()
    
    # Use model's batch translation (or loop if default)
    translated = model.translate_batch(texts, target_lang)
    
    df_trans = df.copy()
    col_name = f'text_{"english" if target_lang == "en" else "chinese"}'
    df_trans[col_name] = translated
    
    print(f"Saving translations to {cache_path}")
    df_trans.to_csv(cache_path, index=False)
    return df_trans

def run_task_b(model_name, train_df, test_df, output_dir, fine_tune=False):
    print("\n" + "="*40)
    print(f"TASK B: Cross-Lingual Bias (Delta-F1) - Model: {model_name}")
    print("="*40)
    
    model = get_model(model_name)
    results = {}
    
    # 1. Native (Spanish)
    print("\n1. Evaluating Native (Spanish)...")
    
    should_train = fine_tune or hasattr(model, 'train')
    
    if should_train:
        model.fine_tune(
            train_df['text_original'].astype(str).tolist(), 
            train_df['label_primary'].tolist()
        )
    
    metrics_es = evaluate_model(model, test_df, 'text_original', 'label_primary')
    f1_es = metrics_es['f1_macro']
    results['es'] = f1_es
    
    # Directory for translations
    trans_dir = os.path.join(output_dir, "translations")
    os.makedirs(trans_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")

    # 2. English
    print("\n2. Evaluating English...")
    train_en = ensure_translations(model, train_df, 'en', os.path.join(trans_dir, f"train_{safe_name}_en.csv"))
    test_en = ensure_translations(model, test_df, 'en', os.path.join(trans_dir, f"test_{safe_name}_en.csv"))
    
    if should_train:
        print("Retraining on English data...")
        model.reset() # Important to reset before retraining on new language
        model.fine_tune(
            train_en['text_english'].astype(str).tolist(),
            train_en['label_primary'].tolist()
        )
        
    metrics_en = evaluate_model(model, test_en, 'text_english', 'label_primary')
    f1_en = metrics_en['f1_macro']
    results['en'] = f1_en
    
    # 3. Chinese
    print("\n3. Evaluating Chinese...")
    train_zh = ensure_translations(model, train_df, 'zh', os.path.join(trans_dir, f"train_{safe_name}_zh.csv"))
    test_zh = ensure_translations(model, test_df, 'zh', os.path.join(trans_dir, f"test_{safe_name}_zh.csv"))
    
    if should_train:
        print("Retraining on Chinese data...")
        model.reset()
        model.fine_tune(
            train_zh['text_chinese'].astype(str).tolist(),
            train_zh['label_primary'].tolist()
        )
        
    metrics_zh = evaluate_model(model, test_zh, 'text_chinese', 'label_primary')
    f1_zh = metrics_zh['f1_macro']
    results['zh'] = f1_zh
    
    # Summary
    delta_en = f1_en - f1_es
    delta_zh = f1_zh - f1_es
    
    print("\n" + "-"*30)
    print(f"Native F1: {f1_es:.4f}")
    print(f"English F1: {f1_en:.4f} (Delta: {delta_en:.4f})")
    print(f"Chinese F1: {f1_zh:.4f} (Delta: {delta_zh:.4f})")
    
    if delta_en > 0:
        print(">> Anglocentric Bias Detected")
    else:
        print(">> Robust / Native Preference")
        
    with open(os.path.join(output_dir, "task_b_results.json"), "w") as f:
        json.dump({
            'f1_es': f1_es, 'f1_en': f1_en, 'f1_zh': f1_zh,
            'delta_en': delta_en, 'delta_zh': delta_zh
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run SANR-Embed Benchmark")
    parser.add_argument("--data_dir", default="SANR-Embed/data/processed", help="Data directory")
    parser.add_argument("--results_dir", default="SANR-Embed/results", help="Results directory")
    parser.add_argument("--task", choices=['A', 'B', 'all'], default='all', help="Task to run")
    parser.add_argument("--model", default="logreg", help="Model to evaluate (must be in registry)")
    
    # New arguments for Fine-Tuning Comparison
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning step before evaluation (if model supports it)")
    parser.add_argument("--k_folds", type=int, default=0, help="Run K-Fold Cross Validation (on training set) instead of standard Train/Test split. Set > 1 to enable.")
    
    args = parser.parse_args()
    
    try:
        train_df, test_df = load_data(args.data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    if args.task in ['A', 'all']:
        run_task_a(args.model, train_df, test_df, args.results_dir, fine_tune=args.fine_tune, k_folds=args.k_folds)
        
    if args.task in ['B', 'all']:
        # Task B doesn't really support K-Fold in the same way cleanly yet (triple language), 
        # so we only pass fine_tune. K-Fold is primarily for Task A (Native) comparison.
        if args.k_folds > 0:
            print("Note: K-Fold CV is currently only implemented for Task A. Running Task B with standard split...")
        run_task_b(args.model, train_df, test_df, args.results_dir, fine_tune=args.fine_tune)
        
    print(f"\nBenchmark run complete. Results in {args.results_dir}")

if __name__ == "__main__":
    main()
