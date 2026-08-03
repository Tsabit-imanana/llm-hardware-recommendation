import os
import shutil
# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

QUANT_CONFIGS = {
    "FP16": [
        ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-f16.gguf"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-fp16-00001-of-00004.gguf", "qwen2.5-7b-instruct-fp16-00002-of-00004.gguf", "qwen2.5-7b-instruct-fp16-00003-of-00004.gguf", "qwen2.5-7b-instruct-fp16-00004-of-00004.gguf"])
    ],
    "Q8_0": [
        ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q8_0.gguf"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf", "qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf", "qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf"])
    ],
    "Q6_K": [
        ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q6_K.gguf"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf", "qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf"])
    ],
    "Q5_K_M": [
        ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q5_K_M.gguf"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf"])
    ],
    "Q4_K_M": [
        ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"])
    ]
}

TARGET_FILENAMES = {
    "FP16": "qwen2.5-7b-instruct-fp16.gguf",
    "Q8_0": "qwen2.5-7b-instruct-q8_0.gguf",
    "Q6_K": "qwen2.5-7b-instruct-q6_k.gguf",
    "Q5_K_M": "qwen2.5-7b-instruct-q5_k_m.gguf",
    "Q4_K_M": "qwen2.5-7b-instruct-q4_k_m.gguf"
}

SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("Starting GGUF model download pipeline...")
    for quant, sources in QUANT_CONFIGS.items():
        target_name = TARGET_FILENAMES[quant]
        target_path = os.path.join(SAVE_DIR, target_name)
        if os.path.exists(target_path):
            print(f"✓ {quant} already exists at {target_path}")
            continue

        print(f"--> Downloading {quant}...")
        downloaded = False
        for repo_id, remote_spec in sources:
            try:
                if isinstance(remote_spec, str):
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=remote_spec,
                        local_dir=SAVE_DIR,
                        local_dir_use_symlinks=False
                    )
                    if os.path.basename(downloaded_path) != target_name:
                        shutil.move(downloaded_path, target_path)
                    downloaded = True
                    print(f"✓ Successfully downloaded {quant} from {repo_id}")
                    break
                elif isinstance(remote_spec, list):
                    shard_paths = []
                    for shard_file in remote_spec:
                        p = hf_hub_download(
                            repo_id=repo_id,
                            filename=shard_file,
                            local_dir=SAVE_DIR,
                            local_dir_use_symlinks=False
                        )
                        shard_paths.append(p)
                    # For sharded files, point target symlink/copy to 00001
                    first_shard = shard_paths[0]
                    if not os.path.exists(target_path):
                        shutil.copy(first_shard, target_path)
                    downloaded = True
                    print(f"✓ Successfully downloaded sharded {quant} from {repo_id}")
                    break
            except Exception as e:
                print(f"  Warning: failed from {repo_id}: {e}")

        if not downloaded:
            print(f"✗ Failed to download {quant} from all available sources.")

    print("All requested models processed.")

if __name__ == "__main__":
    main()
