# Local LLM Quantization Benchmark & Hardware-Aware Recommendation Framework

An end-to-end benchmarking framework and inferential statistical analysis pipeline designed to evaluate the impact of Large Language Model (LLM) quantization levels on code execution accuracy (*Pass Rate / Unitarity*), while providing hardware-aware model quantization recommendations based on available System RAM / GPU VRAM constraints.

---

## 📋 Minimum System Requirements

Ensure your environment meets the following specifications before proceeding:

### Hardware Requirements
* **CPU**: Multi-core processor (x86_64 architecture or Apple Silicon ARM64).
* **System RAM**: Minimum **16 GB RAM** (Recommended: **32 GB+** when evaluating `FP16` unquantized models on CPU).
* **VRAM (Optional but Recommended)**:
  * Minimum **6 GB - 8 GB VRAM** for GPU acceleration on `Q4_K_M` / `Q5_K_M` quantization levels.
  * **16 GB+ VRAM** to offload full `FP16` models to GPU.
* **Disk Space**: At least **40 GB** free storage space (to store multiple `.gguf` model weights).

### Software Requirements
* **Operating System**: Linux (Ubuntu 20.04+, Debian, Arch Linux, RHEL) or macOS (macOS 12+ Monterey / Ventura / Sonoma / Sequoia).
* **Python**: Python `3.10` or `3.11` (Recommended).
* **C++ Compiler & Build Toolchain**:
  * **Linux**: `gcc`, `g++`, `make`, `cmake`.
  * **macOS**: Xcode Command Line Tools (`xcode-select --install`).

---

## ⚙️ Installation & Setup Guide (Linux & macOS)

### 1. Clone Repository & Setup Virtual Environment
Open your terminal and execute the following commands:

```bash
# Clone the repository
git clone https://github.com/username/repository-name.git
cd repository-name

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (Linux / macOS)
source venv/bin/activate
```

---

### 2. Installing `llama-cpp-python` (Hardware Acceleration)

Install `llama-cpp-python` according to your specific hardware setup:

#### A. Linux Users (NVIDIA GPU - CUDA Acceleration)
If you have an NVIDIA GPU with the CUDA Toolkit installed:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

#### B. macOS Users (Apple Silicon M1/M2/M3/M4 - Metal Acceleration)
Metal GPU acceleration is supported natively on Apple Silicon:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

#### C. CPU Only (Linux / macOS without GPU)
For CPU-only execution without GPU offloading:
```bash
pip install llama-cpp-python
```

---

### 3. Install Additional Dependencies
Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Pipeline & Workflow Order

Execute the pipeline scripts in the following order:

```
[Step 1] download_models.py  ➜  [Step 2] eval_real_llm.py  ➜  [Step 3] run_real_stats.py
```

### Step 1: Download GGUF Model Weights
Run the script below to download `Qwen2.5-7B-Instruct` model weights across 5 quantization levels (`FP16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`) from Hugging Face into the `./models/` directory:

```bash
python scripts/download_models.py
```
*Note: Download time depends on your internet speed (total size: ~39 GB).*

---

### Step 2: Run Real LLM Evaluation & Sandboxed Code Execution
Run the LLM inference evaluation and unit test execution inside the isolated sandbox:

```bash
python eval_real_llm.py
```
* **What it does**: Loads each model variant from `./models/`, generates Python code for HumanEval prompts, and validates execution inside a sandboxed environment.
* **Output**: Evaluation pass-rate matrix is saved to `./results/real_eval_matrix.json`.

> [!TIP]
> For a quick trial run, edit `eval_real_llm.py` and change `NUM_PROMPTS = 20` to `NUM_PROMPTS = 2`.

---

### Step 3: Run Inferential Statistical Engine & Recommendation Matrix
Once `real_eval_matrix.json` is generated, run the statistical analysis and hardware recommendation pipeline:

```bash
python run_real_stats.py
```
* **Terminal Outputs**:
  * **Descriptive Statistics**: Mean, Std Dev, Min, Max pass rate per quantization level.
  * **Inferential Statistical Tests**: *Friedman Chi-Squared ($\chi^2$)*, *p-value*, *Kendall's W* (Effect Size), and *Degradation Elbow Point*.
  * **Post-Hoc Pairwise Tests**: *Dunn's Test* matrix with Bonferroni adjustment.
  * **Hardware Recommendation Matrix**: Optimal model level recommendation for target VRAM thresholds (16GB, 8GB, 6GB, 5GB, 4GB).

---

### 💡 (Optional) Synthetic Simulation Mode
To test the entire statistical pipeline and sandboxed engine instantly without downloading physical model weights:

```bash
python main.py
```

---

## 🛠️ Model & Quantization Customization Guide

To evaluate different model families (e.g., `Llama-3.1-8B-Instruct`) or alternative quantization levels (e.g., `Q3_K_M`):

1. **Modify `scripts/download_models.py`**:
   * Update the Hugging Face repository ID and `.gguf` filenames in `QUANT_CONFIGS`.
   * Update local target names in `TARGET_FILENAMES`.
2. **Modify `eval_real_llm.py`**:
   * Update `MODELS_MAP` paths to match the new filenames in `./models/`.

---

## 📁 Project Directory Structure

```text
.
├── core/
│   ├── execution_engine.py   # Isolated code execution sandbox
│   └── recommender.py        # Hardware-aware VRAM recommendation engine
├── data/
│   └── dataset_loader.py     # HumanEval benchmark dataset loader
├── models/                   # Directory storing GGUF model weights
├── results/
│   └── real_eval_matrix.json # Generated evaluation score matrix
├── scripts/
│   └── download_models.py    # Automated Hugging Face GGUF downloader
├── stats/
│   └── stat_engine.py        # Inferential statistical engine (Friedman, Kendall, Dunn)
├── eval_real_llm.py          # Primary real LLM evaluation pipeline script
├── run_real_stats.py         # Primary statistical & recommendation execution script
├── main.py                   # End-to-end synthetic simulation prototype script
├── requirements.txt          # Python package dependencies
└── README.md                 # Framework documentation
```

---

## 📜 License
This project is open-source and intended for research on LLM quantization performance trade-offs. Feel free to modify and build upon it.
