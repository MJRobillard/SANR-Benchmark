from huggingface_hub import snapshot_download
import os

def download_qwen25():
    print("Downloading Qwen-2.5...")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_dir=os.path.expanduser("~/.cache/sanr_models/qwen25"),
        ignore_patterns=["*.md", "*.png", "*.txt"]
    )

if __name__ == "__main__":
    download_qwen25()





