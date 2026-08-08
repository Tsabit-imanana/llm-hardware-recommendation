import os
import sys
import json
import argparse
import time
# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hf_helper import DownloadProgressLogger

MODELS_DIR = "./models"

class TeeLogger:
    """Redirects stdout to both terminal screen and log file."""
    def __init__(self, log_filepath):
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger("./logs/download_execution.log")

def main():
    parser = argparse.ArgumentParser(description="Download multiple quantized GGUF models from a HF repo.")
    parser.add_argument("--repo", type=str, help="Hugging Face repository ID")
    parser.add_argument("--config", type=str, help="JSON file path containing repo and file list")
    parser.add_argument("files", nargs="*", help="List of GGUF filenames to download")

    args = parser.parse_args()

    repo_id = args.repo
    files_to_download = args.files

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            repo_id = cfg.get("repo_id", repo_id)
            files_to_download = cfg.get("files", files_to_download)

    if not repo_id or not files_to_download:
        print("Error: Missing repository ID or file list to download.", flush=True)
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    total_files = len(files_to_download)
    print(f"\n==========================================", flush=True)
    print(f" 📥 BATCH DOWNLOADING {total_files} QUANTIZED MODELS", flush=True)
    print(f" Repository: {repo_id}", flush=True)
    print(f" Target Directory: {MODELS_DIR}", flush=True)
    print(f"==========================================\n", flush=True)

    for idx, fname in enumerate(files_to_download, 1):
        print(f"\n[{idx}/{total_files}] STARTING DOWNLOAD: {fname}", flush=True)
        t0 = time.time()
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
                tqdm_class=DownloadProgressLogger
            )
            elapsed = time.time() - t0
            size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
            print(f"✓ [{idx}/{total_files}] Downloaded {fname} ({size_mb:.2f} MB) in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"❌ [{idx}/{total_files}] Failed to download {fname}: {e}", flush=True)

    print("\n==========================================", flush=True)
    print(" 🎉 Batch Model Quantization Download Complete!", flush=True)
    print("==========================================", flush=True)

if __name__ == "__main__":
    main()
