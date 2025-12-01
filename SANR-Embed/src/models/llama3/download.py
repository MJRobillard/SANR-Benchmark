from huggingface_hub import snapshot_download
import os

def download_llama3():
    print("Downloading Llama-3...")
    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        local_dir=os.path.expanduser("~/.cache/sanr_models/llama3"),
        ignore_patterns=["*.md", "*.png", "*.txt"]
    )

if __name__ == "__main__":
    download_llama3()





