import os
import shutil
import time
# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi, hf_hub_download
# pyrefly: ignore [missing-import]
from tqdm.auto import tqdm

class DownloadProgressLogger(tqdm):
    """
    Custom tqdm logger that prints download percentage and byte counts to stdout on separate lines.
    This allows process execution runners and stream log parsers to extract real-time download progress.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_pct = -1

    def update(self, n=1):
        res = super().update(n)
        if self.total and self.total > 0:
            pct = int((self.n / self.total) * 100)
            if pct != self._last_pct and (pct % 2 == 0 or pct == 100):
                self._last_pct = pct
                mb_n = self.n / (1024 * 1024)
                mb_total = self.total / (1024 * 1024)
                desc = self.desc or "Downloading"
                print(f"PROGRESS: [{desc}] {pct}% ({mb_n:.1f} MB / {mb_total:.1f} MB)", flush=True)
        return res

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_GGUF_MODELS = [
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "TheBloke/Llama-2-7B-Chat-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "city96/ComfyUI-GGUF"
]

def search_hf_models(query: str = "", limit: int = 25):
    """
    Search Hugging Face models using HfApi.
    Supports keywords, direct repository IDs ('owner/repo'), and full Hugging Face URLs.
    """
    api = HfApi()
    try:
        query = query.strip()
        if not query:
            # Return popular default GGUF model repositories
            return [{"id": repo, "downloads": 10000, "likes": 500} for repo in DEFAULT_GGUF_MODELS]

        # Strip URL prefix if user pasted a Hugging Face URL
        if "huggingface.co/" in query:
            query = query.split("huggingface.co/")[-1].strip("/")
            parts = query.split("/")
            if len(parts) >= 2:
                query = f"{parts[0]}/{parts[1]}"

        results = []
        seen_ids = set()

        # If query is a direct repo ID (owner/repo), fetch its exact info first
        if "/" in query and len(query.split("/")) == 2:
            try:
                info = api.model_info(query)
                results.append({
                    "id": info.id,
                    "downloads": getattr(info, "downloads", 0),
                    "likes": getattr(info, "likes", 0),
                    "tags": getattr(info, "tags", [])
                })
                seen_ids.add(info.id)
            except Exception:
                pass

        models = api.list_models(search=query, limit=limit, sort="downloads")
        for m in models:
            if m.id not in seen_ids:
                results.append({
                    "id": m.id,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "tags": getattr(m, "tags", [])
                })
                seen_ids.add(m.id)
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

def list_repo_gguf_files_info(repo_id: str):
    """
    Lists all .gguf files inside a Hugging Face repository along with size info.
    """
    api = HfApi()
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        siblings = getattr(info, "siblings", [])
        results = []
        for s in siblings:
            rfilename = getattr(s, "rfilename", "")
            if rfilename.endswith(".gguf"):
                size_bytes = getattr(s, "size", 0) or 0
                size_gb = size_bytes / (1024**3)
                results.append({
                    "filename": rfilename,
                    "size_gb": round(size_gb, 2) if size_gb > 0 else "Unknown",
                    "size_bytes": size_bytes
                })
        if not results:
            # Fallback if files_metadata didn't include siblings sizes
            filenames = list_repo_gguf_files(repo_id)
            results = [{"filename": f, "size_gb": "Unknown", "size_bytes": 0} for f in filenames]
        return results
    except Exception as e:
        print(f"Error fetching file info for {repo_id}: {e}")
        filenames = list_repo_gguf_files(repo_id)
        return [{"filename": f, "size_gb": "Unknown", "size_bytes": 0} for f in filenames]

def get_quant_label(filename: str) -> str:
    """Helper to derive clean quantization label from model filename."""
    fn_lower = filename.lower()
    quant_patterns = [
        "fp16", "f32", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0", "q4_k_m", "q4_k_s", "q4_0", 
        "q3_k_l", "q3_k_m", "q3_k_s", "q2_k", 
        "iq4_xs", "iq4_nl", "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq1_s", "iq1_m"
    ]
    for q in quant_patterns:
        if q in fn_lower:
            return q.upper()
    clean = os.path.splitext(filename)[0]
    return clean.replace(" ", "_")

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
            local_dir_use_symlinks=False,
            tqdm_class=DownloadProgressLogger
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
