import pandas as pd
import argparse
import os
import time
from tqdm import tqdm
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

def translate_text(text_list, model_name, src_lang=None, tgt_lang=None, batch_size=8, mock=False):
    """
    Translates a list of texts using a Hugging Face model, Google Translate (via deep-translator), or returns mock data.
    Supports standard translation pipelines and NLLB multilingual pipelines.
    """
    if mock:
        return [f"[Translated to {tgt_lang or 'target'}] {str(t)[:20]}..." for t in text_list]
    
    # --- Google Translate Support (deep-translator) ---
    if model_name.lower() == "google":
        try:
            from deep_translator import GoogleTranslator
            
            # Map standard codes to deep-translator codes
            g_src = src_lang if src_lang else 'auto'
            g_dest = tgt_lang if tgt_lang else 'en'
            
            results = []
            print(f"Translating via Google Translate ({g_src} -> {g_dest})...")
            
            # Initialize translator
            translator = GoogleTranslator(source=g_src, target=g_dest)
            
            # Process one by one or batch if library supports it well (deep-translator has batch)
            # But batch might be limited by URL length. Safe to do small batches or iterative.
            # deep_translator's translate_batch is handy.
            
            # Using translate_batch for efficiency
            for i in tqdm(range(0, len(text_list), batch_size), desc="Google Translate"):
                batch = text_list[i:i+batch_size]
                valid_batch = [t if isinstance(t, str) and t.strip() else "" for t in batch]
                
                try:
                    # Add jitter/delay to avoid IP ban
                    time.sleep(random.uniform(0.2, 0.8))
                    batch_res = translator.translate_batch(valid_batch)
                    results.extend(batch_res)
                except Exception as e:
                    print(f"Google Translate error in batch {i}: {e}")
                    # Fallback one by one if batch fails
                    for t in valid_batch:
                        try:
                            results.append(translator.translate(t))
                        except:
                            results.append("[Error]")
            
            return results
            
        except ImportError:
            print("deep-translator library not found. Please install it: pip install deep-translator")
            return [f"[Missing deep-translator] {str(t)[:20]}..." for t in text_list]
    
    # --- Hugging Face Support ---
    try:
        from transformers import pipeline
    except ImportError:
        print("Transformers library not found. Please install it.")
        return []

    print(f"Loading translation model: {model_name}...")
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("TORCH_DEVICE") == "cuda" else -1
    
    try:
        # Configure pipeline arguments based on model type
        pipeline_args = {
            "task": "translation",
            "model": model_name,
            "device": device
        }
        
        # NLLB specific configuration
        if "nllb" in model_name.lower():
            if not src_lang or not tgt_lang:
                raise ValueError("NLLB models require src_lang and tgt_lang arguments (e.g. 'spa_Latn', 'eng_Latn')")
            pipeline_args["src_lang"] = src_lang
            pipeline_args["tgt_lang"] = tgt_lang
            print(f"Configured NLLB: {src_lang} -> {tgt_lang}")

        translator = pipeline(**pipeline_args)
        
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        print("Falling back to mock translation.")
        return [f"[Translated] {str(t)[:20]}..." for t in text_list]
    
    results = []
    # Process in batches
    for i in tqdm(range(0, len(text_list), batch_size), desc=f"Translating with {model_name}"):
        batch = text_list[i:i+batch_size]
        
        valid_indices = [j for j, t in enumerate(batch) if isinstance(t, str) and t.strip()]
        clean_batch = [batch[j] for j in valid_indices]
        
        batch_results = [""] * len(batch)
        
        if clean_batch:
            try:
                # Truncate to model max length usually 512
                # NLLB might behave differently, but pipeline usually handles it via tokenizer
                if "nllb" in model_name.lower():
                    # NLLB pipeline needs explicit src_lang/tgt_lang in call sometimes depending on version
                    # But usually initialized in pipeline is enough.
                    # Newer transformers might strictly require it in the call if not set in config.
                    # We will pass max_length just in case.
                    outputs = translator(clean_batch, max_length=512)
                else:
                    outputs = translator(clean_batch, truncation=True, max_length=512)
                    
                translated_texts = [o['translation_text'] for o in outputs]
                
                for j, valid_idx in enumerate(valid_indices):
                    batch_results[valid_idx] = translated_texts[j]
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Fallback for this batch
                for j in valid_indices:
                    batch_results[j] = "[Error]"
        
        results.extend(batch_results)
            
    return results

def translate_and_save(input_path, model_name, output_path=None, mock=False, batch_size=8):
    """
    Translates a specific CSV file using the specified model and saves it.
    Generates model-specific filenames if output_path is not provided.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records.")
    
    if 'text_original' not in df.columns:
        print("Column 'text_original' missing.")
        return None

    # Clean model name for filename
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{safe_model_name}_translated{ext}"

    # Determine Model Configuration
    # NLLB Codes: Spanish (spa_Latn), English (eng_Latn), Chinese Simplified (zho_Hans)
    if "nllb" in model_name.lower():
        en_model = model_name
        en_src = "spa_Latn"
        en_tgt = "eng_Latn"
        
        zh_model = model_name
        zh_src = "spa_Latn"
        zh_tgt = "zho_Hans"
    elif model_name.lower() == "google":
        en_model = "google"
        en_src = "es"
        en_tgt = "en"
        
        zh_model = "google"
        zh_src = "es"
        zh_tgt = "zh-CN" # deep-translator expects zh-CN or zh-TW
    else:
        # Fallback / Other models
        if model_name == "opus":
            en_model = "Helsinki-NLP/opus-mt-es-en"
            zh_model = "Helsinki-NLP/opus-mt-es-zh"
        else:
            # Custom model path
            en_model = model_name
            zh_model = model_name 
            
        en_src = None; en_tgt = None
        zh_src = None; zh_tgt = None

    # Translate to English
    print(f"\n--- Translating to English ({model_name}) ---")
    df['text_english'] = translate_text(
        df['text_original'].tolist(), 
        model_name=en_model,
        src_lang=en_src,
        tgt_lang=en_tgt,
        batch_size=batch_size,
        mock=mock
    )
    
    # Translate to Chinese
    print(f"\n--- Translating to Chinese ({model_name}) ---")
    df['text_chinese'] = translate_text(
        df['text_original'].tolist(), 
        model_name=zh_model,
        src_lang=zh_src,
        tgt_lang=zh_tgt,
        batch_size=batch_size,
        mock=mock
    )
    
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Translate SANR-Embed dataset for Task B")
    parser.add_argument("--input", default="SANR-Embed/data/processed/gold_standard.csv", help="Input CSV path")
    parser.add_argument("--output", help="Output CSV path (optional, auto-generated if skipped)")
    parser.add_argument("--mock", action="store_true", help="Use mock translation for testing/speed")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M", 
                        help="Model name. Use 'google' for Google Translate, 'opus' for Helsinki-NLP.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for translation")
    args = parser.parse_args()
    
    translate_and_save(
        input_path=args.input, 
        model_name=args.model, 
        output_path=args.output, 
        mock=args.mock, 
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
