import os
import sys
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
        quant_match = re.search(r"Running Evaluation for Quantization:\s*([A-Za-z0-9_]+)", line)
        if quant_match:
            self.current_quant = quant_match.group(1)
            self.current_status = f"Evaluating Quantization: {self.current_quant}"
            return

        # Prompt progress detection: [Q8_0] Prompt 5/20 (HumanEval/0) generating code...
        prompt_match = re.search(r"\[([A-Za-z0-9_]+)\]\s+Prompt\s+(\d+)/(\d+)", line)
        if prompt_match:
            self.current_quant = prompt_match.group(1)
            curr = int(prompt_match.group(2))
            total = int(prompt_match.group(3))
            self.current_prompt_index = curr
            self.total_prompts = total
            
            # Map quantization to step fraction
            quants = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
            q_idx = quants.index(self.current_quant) if self.current_quant in quants else 0
            base_p = q_idx / len(quants)
            within_p = (curr / total) * (1.0 / len(quants))
            self.progress_percentage = min(0.99, base_p + within_p)
            self.current_status = f"Evaluating {self.current_quant} - Prompt {curr}/{total}"
            return

        # Download detection: Downloading FP16... or Successfully downloaded
        if "Downloading " in line:
            self.current_status = line.strip()
        elif "✓" in line or "✗" in line:
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
