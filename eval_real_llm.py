import os
import sys
import json
import pandas as pd
# pyrefly: ignore [missing-import]
from llama_cpp import Llama
from data.dataset_loader import load_humaneval_plus_subset
from core.execution_engine import run_code_in_sandbox

class TeeLogger:
    """Redirects stdout to both terminal screen and a log file."""
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

sys.stdout = TeeLogger("./logs/real_eval_execution.log")


DEFAULT_MODELS_MAP = {
    "FP16": "./models/qwen2.5-7b-instruct-fp16.gguf",
    "Q8_0": "./models/qwen2.5-7b-instruct-q8_0.gguf",
    "Q6_K": "./models/qwen2.5-7b-instruct-q6_k.gguf",
    "Q5_K_M": "./models/qwen2.5-7b-instruct-q5_k_m.gguf",
    "Q4_K_M": "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
}

# Locked Control Variables (Defaults)
SEED = 42
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_TOKENS = 512
DEFAULT_NUM_PROMPTS = 20

def load_eval_config():
    config_file = "./results/eval_config.json"
    models_map = DEFAULT_MODELS_MAP.copy()
    num_prompts = DEFAULT_NUM_PROMPTS

    # Detect system GPU hardware
    try:
        from utils.gpu_helper import detect_gpu_hardware, configure_amd_gpu_env
        hw_info = detect_gpu_hardware()
        print(f"Hardware Detected: {hw_info['vendor']} ({hw_info['gpu_name']}) | Recommended Backend: {hw_info['recommended_backend']} | VRAM Free: {hw_info['vram_free_gb']} GB", flush=True)
    except Exception:
        hw_info = {}

    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "models_map" in cfg and cfg["models_map"]:
                    models_map = cfg["models_map"]
                if "num_prompts" in cfg and cfg["num_prompts"]:
                    num_prompts = int(cfg["num_prompts"])
                if "hsa_override_gfx_version" in cfg:
                    gfx_ver = str(cfg["hsa_override_gfx_version"])
                    if gfx_ver.strip():
                        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_ver.strip()
                        print(f"Configured AMD HSA_OVERRIDE_GFX_VERSION = {gfx_ver.strip()}", flush=True)
                if "hip_visible_devices" in cfg:
                    hip_dev = str(cfg["hip_visible_devices"])
                    os.environ["HIP_VISIBLE_DEVICES"] = hip_dev
                    print(f"Configured AMD HIP_VISIBLE_DEVICES = {hip_dev}", flush=True)
            print(f"Loaded custom evaluation config from {config_file}: {len(models_map)} models, {num_prompts} prompts.", flush=True)
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}. Using defaults.", flush=True)

    return models_map, num_prompts

def extract_python_code(raw_response: str) -> str:
    """Extracts raw executable code block from Markdown output if present."""
    if "```python" in raw_response:
        return raw_response.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_response:
        return raw_response.split("```")[1].split("```")[0].strip()
    return raw_response.strip()

def run_evaluation():
    models_map, num_prompts = load_eval_config()
    dataset = load_humaneval_plus_subset(limit=num_prompts)
    results_matrix = {quant: [] for quant in models_map.keys()}
    
    for quant_level, model_path in models_map.items():

        if not os.path.exists(model_path):
            print(f"Skipping {quant_level}: Model file not found at {model_path}", flush=True)
            continue
            
        print(f"\n==========================================", flush=True)
        print(f" Running Evaluation for Quantization: {quant_level}", flush=True)
        print(f"==========================================", flush=True)
        print(f"--> Loading model weights: {model_path} ...", flush=True)
        
        # Instantiate low-level llama.cpp engine
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Offload layers to GPU if CUDA available
            seed=SEED,
            verbose=True
        )
        print(f"✓ Model {quant_level} loaded into memory successfully.", flush=True)
        
        for idx, item in enumerate(dataset):
            print(f"[{quant_level}] Prompt {idx+1}/{num_prompts} ({item['task_id']}) generating code...", flush=True)
            prompt_text = f"Complete the following Python function. Output ONLY executable Python code inside codeblocks:\n\n{item['prompt']}"
            
            # Deterministic Inference Execution
            output = llm(
                prompt=prompt_text,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=["\n\n\n", "Problem:", "Note:"]
            )
            
            raw_gen = output["choices"][0]["text"]
            gen_code = extract_python_code(raw_gen)
            
            # Reconstruct full script
            full_code = f"{item['prompt']}\n{gen_code}"
            assertions = [item["test_code"]]
            
            # Execute in sandbox using existing execution engine
            eval_res = run_code_in_sandbox(full_code, assertions, timeout_sec=5.0)
            pass_rate = eval_res["unit_test_pass_rate"]
            
            results_matrix[quant_level].append(pass_rate)
            print(f"[{quant_level}] Prompt {idx+1}/{num_prompts} ({item['task_id']}) -> UR: {pass_rate:.2f}", flush=True)

    # Save real matrix output to JSON
    os.makedirs("./results", exist_ok=True)
    with open("./results/real_eval_matrix.json", "w") as f:
        json.dump(results_matrix, f, indent=4)
        
    print("\n✓ Real LLM Evaluation Completed! Saved matrix to ./results/real_eval_matrix.json", flush=True)

if __name__ == "__main__":
    run_evaluation()
