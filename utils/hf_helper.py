import os
import shutil
import time
# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi, hf_hub_download

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_GGUF_MODELS = [
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "TheBloke/Llama-2-7B-Chat-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "city96/ComfyUI-GGUF"
]

def search_hf_models(query: str = "", limit: int = 15):
    """
    Search Hugging Face models using HfApi.
    """
    api = HfApi()
    try:
        if not query.strip():
            # Return popular default GGUF model repositories
            return [{"id": repo, "downloads": 10000, "likes": 500} for repo in DEFAULT_GGUF_MODELS]
        
        models = api.list_models(search=query, limit=limit, sort="downloads")
        results = []
        for m in models:
            results.append({
                "id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "tags": getattr(m, "tags", [])
            })
        return results
    except Exception as e:
        print(f"Error querying Hugging Face API: {e}")
        return [{"id": repo, "downloads": 0, "likes": 0} for repo in DEFAULT_GGUF_MODELS]

def list_repo_gguf_files(repo_id: str):
    """
    Lists all .gguf files inside a Hugging Face repository.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]
        return gguf_files
    except Exception as e:
        print(f"Error fetching files for {repo_id}: {e}")
        return []

def get_local_models():
    """
    Lists existing downloaded model files in ./models directory.
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    local_files = []
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".gguf") or f.endswith(".bin"):
                path = os.path.join(root, f)
                size_bytes = os.path.getsize(path)
                size_mb = size_bytes / (1024 * 1024)
                size_gb = size_mb / 1024
                local_files.append({
                    "filename": f,
                    "rel_path": os.path.relpath(path, start="./"),
                    "abs_path": path,
                    "size_mb": round(size_mb, 2),
                    "size_gb": round(size_gb, 2),
                    "mod_time": time.ctime(os.path.getmtime(path))
                })
    return sorted(local_files, key=lambda x: x["filename"])

def download_hf_file(repo_id: str, filename: str, custom_name: str = None, progress_callback=None):
    """
    Downloads a specific model file from HF repo to ./models.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_filename = custom_name if custom_name else filename
    target_path = os.path.join(MODELS_DIR, target_filename)

    if progress_callback:
        progress_callback(0.1, f"Initiating download for {filename} from {repo_id}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        
        if progress_callback:
            progress_callback(0.9, f"Downloaded to cache. Processing file location...")

        if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
            shutil.move(downloaded_path, target_path)

        if progress_callback:
            progress_callback(1.0, f"Successfully saved {target_filename} ({os.path.getsize(target_path) / (1024*1024):.2f} MB)")
            
        return True, target_path, "Download complete."
    except Exception as e:
        err_msg = str(e)
        if progress_callback:
            progress_callback(1.0, f"Failed: {err_msg}")
        return False, None, err_msg
