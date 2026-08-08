import os
import sys
import json
import subprocess
import threading
import queue
import time
import re

class BackgroundTaskRunner:
    def __init__(self):
        self.process = None
        self.log_lines = []
        self.is_running = False
        self.exit_code = None
        self.lock = threading.Lock()
        self.progress_percentage = 0.0
        self.current_status = "Idle"
        self.current_quant = None
        self.current_prompt_index = 0
        self.total_prompts = 20

    def start_task(self, command_args, cwd=None):
        with self.lock:
            if self.is_running:
                return False, "Task is already running."
            
            self.log_lines = []
            self.is_running = True
            self.exit_code = None
            self.progress_percentage = 0.0
            self.current_status = "Starting process..."
            self.current_quant = None
            self.current_prompt_index = 0

            # Launch process with unbuffered output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd or os.getcwd(),
                env=env
            )

            # Thread to read stdout line-by-line
            thread = threading.Thread(target=self._read_output)
            thread.daemon = True
            thread.start()
            return True, "Process started."

    def _read_output(self):
        for line in iter(self.process.stdout.readline, ''):
            clean_line = line.rstrip('\r\n')
            if clean_line:
                with self.lock:
                    self.log_lines.append(clean_line)
                    self._parse_line(clean_line)
        
        self.process.stdout.close()
        self.process.wait()

        with self.lock:
            self.exit_code = self.process.returncode
            self.is_running = False
            if self.exit_code == 0:
                self.progress_percentage = 1.0
                self.current_status = "Completed Successfully"
            else:
                self.current_status = f"Failed with exit code {self.exit_code}"

    def _parse_line(self, line: str):
        """Parse log lines to extract real-time progress for eval and download scripts."""
        # Quant level detection: Running Evaluation for Quantization: Q8_0
        quant_match = re.search(r"Running Evaluation for Quantization:\s*(.+)$", line)
        if quant_match:
            self.current_quant = quant_match.group(1).strip()
            self.current_status = f"Evaluating Quantization: {self.current_quant}"
            return

        # Prompt progress detection: [Q8_0] Prompt 5/20 (HumanEval/0) generating code...
        prompt_match = re.search(r"\[(.+?)\]\s+Prompt\s+(\d+)/(\d+)", line)
        if prompt_match:
            self.current_quant = prompt_match.group(1).strip()
            curr = int(prompt_match.group(2))
            total = int(prompt_match.group(3))
            self.current_prompt_index = curr
            self.total_prompts = total
            
            # Read total models count from eval_config.json if available
            total_models = 5
            q_idx = 0
            try:
                cfg_path = "./results/eval_config.json"
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        if "models_map" in cfg and cfg["models_map"]:
                            models_keys = list(cfg["models_map"].keys())
                            total_models = max(len(models_keys), 1)
                            if self.current_quant in models_keys:
                                q_idx = models_keys.index(self.current_quant)
            except Exception:
                q_idx = 0

            base_p = q_idx / total_models
            within_p = (curr / total) * (1.0 / total_models)
            self.progress_percentage = min(0.99, base_p + within_p)
            self.current_status = f"Evaluating {self.current_quant} - Prompt {curr}/{total}"
            return

        # Download percentage progress detection: PROGRESS: [qwen2.5-7b-instruct-q4_k_m.gguf] 45% (1980.0 MB / 4400.0 MB)
        progress_match = re.search(r"PROGRESS:\s*\[(.*?)\]\s*(\d+)%\s*\((.*?)\)", line)
        if progress_match:
            fname = progress_match.group(1)
            pct_val = int(progress_match.group(2))
            details = progress_match.group(3)
            self.progress_percentage = pct_val / 100.0
            self.current_status = f"Downloading {fname}: {pct_val}% ({details})"
            return

        # Batch item detection: [1/5] STARTING DOWNLOAD: filename.gguf
        batch_match = re.search(r"\[(\d+)/(\d+)\]\s+STARTING DOWNLOAD:\s*(.*)", line)
        if batch_match:
            idx = int(batch_match.group(1))
            total = int(batch_match.group(2))
            fname = batch_match.group(3).strip()
            self.progress_percentage = (idx - 1) / total
            self.current_status = f"[{idx}/{total}] Preparing download for {fname}..."
            return

        # General download status line
        if "Downloading " in line or "STARTING DOWNLOAD" in line:
            self.current_status = line.strip()
        elif "✓" in line or "❌" in line or "✗" in line:
            self.current_status = line.strip()

    def stop_task(self):
        with self.lock:
            if self.process and self.is_running:
                self.process.terminate()
                self.is_running = False
                self.current_status = "Terminated by user"
                return True, "Process terminated."
            return False, "No active process running."

    def get_state(self):
        with self.lock:
            return {
                "is_running": self.is_running,
                "exit_code": self.exit_code,
                "logs": list(self.log_lines),
                "status": self.current_status,
                "progress": self.progress_percentage,
                "quant": self.current_quant,
                "prompt_idx": self.current_prompt_index,
                "total_prompts": self.total_prompts
            }

# Global instances for background runners
eval_runner = BackgroundTaskRunner()
stats_runner = BackgroundTaskRunner()
download_runner = BackgroundTaskRunner()
