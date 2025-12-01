import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.deepseek_v3.download import download_deepseek_v3
from src.models.llama3.download import download_llama3
from src.models.qwen25.download import download_qwen25

def main():
    print("Starting download of all Tier 2 models...")
    try:
        download_deepseek_v3()
    except Exception as e:
        print(f"Failed to download DeepSeek-V3: {e}")

    try:
        download_llama3()
    except Exception as e:
        print(f"Failed to download Llama-3: {e}")

    try:
        download_qwen25()
    except Exception as e:
        print(f"Failed to download Qwen-2.5: {e}")
        
    print("Download process finished.")

if __name__ == "__main__":
    main()





