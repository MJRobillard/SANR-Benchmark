from huggingface_hub import snapshot_download
import os

def download_deepseek_v3():
    print("Downloading DeepSeek-V3...")
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-V3",
        local_dir=os.path.expanduser("~/.cache/sanr_models/deepseek_v3"),
        ignore_patterns=["*.md", "*.png", "*.txt"]
    )

if __name__ == "__main__":
    download_deepseek_v3()

