# Local LLM Quantization Benchmark & Hardware-Aware Recommendation Framework

An end-to-end benchmarking framework and inferential statistical analysis pipeline designed to evaluate the impact of Large Language Model (LLM) quantization levels on code execution accuracy (*Pass Rate / Unitarity*) using **HumanEvalPlus (EvalPlus)**, while providing hardware-aware model quantization recommendations based on available System RAM / GPU VRAM constraints across **NVIDIA CUDA**, **AMD ROCm/HIP**, and **Vulkan** hardware backends.

---

## 📋 Minimum System Requirements

Ensure your environment meets the following specifications before proceeding:

### Hardware Requirements
* **CPU**: Multi-core processor (x86_64 architecture or Apple Silicon ARM64).
* **System RAM**: Minimum **16 GB RAM** (Recommended: **32 GB+** when evaluating `FP16` unquantized models on CPU).
* **VRAM (Optional but Recommended)**:
  * Minimum **6 GB - 8 GB VRAM** for GPU acceleration on `Q4_K_M` / `Q5_K_M` quantization levels.
  * **16 GB+ VRAM** to offload full `FP16` models to GPU.
* **GPU Hardware Backends**:
  * **NVIDIA GPU**: CUDA Toolkit 11.8+ / 12.0+.
  * **AMD GPU**: AMD ROCm 5.x / 6.x (HIP) or Vulkan driver for consumer Radeon RX series.
  * **Apple Silicon**: macOS Metal API.
* **Disk Space**: At least **40 GB** free storage space (to store `.gguf` model weights).

### Software Requirements
* **Operating System**: Linux (Ubuntu 20.04+, Debian, Arch Linux, RHEL) or macOS (macOS 12+).
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
git clone https://github.com/Tsabit-imanana/llm-hardware-recommendation.git
cd llm-hardware-recommendation

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (Linux / macOS)
source venv/bin/activate
```

---

### 2. Installing `llama-cpp-python` (Hardware Acceleration)

Install `llama-cpp-python` according to your specific hardware setup:

#### A. Linux Users (NVIDIA GPU - CUDA Acceleration)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

#### B. Linux Users (AMD GPU - ROCm / HIP Acceleration)
```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" HSA_OVERRIDE_GFX_VERSION=10.3.0 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
*Note: For AMD Radeon RX 6000 series (RDNA2), use `HSA_OVERRIDE_GFX_VERSION=10.3.0`. For RX 7000 series (RDNA3), use `11.0.0`.*

#### C. Universal AMD / Multi-Vendor GPU (Vulkan Acceleration)
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

#### D. macOS Users (Apple Silicon M1/M2/M3/M4 - Metal Acceleration)
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

#### E. CPU Only (Linux / macOS without GPU)
```bash
pip install llama-cpp-python
```

---

### 3. Install Framework Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🌐 Interactive Streamlit Web UI (Real-Time Progress & Analytics)

An interactive, dark-themed Streamlit Web Dashboard is included to manage model downloading, configure AMD/NVIDIA GPU hardware parameters, execute LLM evaluation benchmarks, monitor real-time execution progress, and visualize statistical recommendation matrices.

### Launching the Web UI

Run the following command from your terminal:

```bash
streamlit run app.py
```
*(Or `./venv/bin/streamlit run app.py`)*

Once started, navigate to `http://localhost:8501` in your browser.

### Key Web UI Features

1. **🔍 Project Overview & Workspace Scanner**:
   * Scans python codebase files, total lines of code, downloaded GGUF model files, matrix status (`results/real_eval_matrix.json`), and execution log sizes.
   * Auto-detects multi-vendor GPU hardware (NVIDIA CUDA, AMD ROCm/HIP, Vulkan, CPU) and available VRAM.

2. **📥 Hugging Face Model Downloader (Live Search)**:
   * **Preset Quantization Downloads**: One-click download presets for `FP16`, `Q8_0`, `Q6_K`, `Q5_K_M`, and `Q4_K_M`.
   * **Live Hugging Face Search**: Search Hugging Face Hub repositories live (e.g. `Qwen2.5-7B`, `Llama-3.1`), inspect `.gguf` quantization files, and trigger batch downloads into `./models/`.

3. **🖥️ GPU & Hardware Config (AMD / NVIDIA)**:
   * **Hardware Diagnostics**: Displays detected GPU card model, total/free VRAM, driver version, and recommended acceleration backend.
   * **AMD Driver Settings**: Interactive selector for AMD `HSA_OVERRIDE_GFX_VERSION` (`10.3.0`, `11.0.0`, `9.0.0`) and `HIP_VISIBLE_DEVICES`.
   * **Build Guide**: Embedded compilation and environment setup guide for ROCm and Vulkan.

4. **⚡ LLM Evaluation Engine (`eval_real_llm`)**:
   * Asynchronous non-blocking execution runner on **HumanEvalPlus (EvalPlus)** benchmark prompts with $80\times$ expanded test coverage.
   * **Real-Time Progress Bar**: Tracks active prompt progress (`Prompt X/20`) and quantization level steps (`FP16` → `Q8_0` → `Q6_K` → `Q5_K_M` → `Q4_K_M`).
   * **Live Output Console**: Live log streaming terminal window watching `./logs/real_eval_execution.log`.
   * **Live Metric Charts**: Dynamic Plotly bar charts comparing average unit test pass rates (UR) and per-prompt HumanEvalPlus performance charts.

5. **📊 Statistical Summary & Hardware Recommender (`run_real_stat`)**:
   * **Statistical Test Dashboard**: Instant visual cards for *Friedman Chi-Squared ($\chi^2$)*, *$p$-value*, *Kendall's W Effect Size*, and *Degradation Elbow Point*.
   * **Dunn's Post-Hoc Heatmap**: Interactive Plotly heatmap displaying Bonferroni-adjusted pairwise $p$-values.
   * **Interactive Hardware Recommendation Engine**: Dynamic VRAM slider ($1.0\text{ GB} - 32.0\text{ GB}$) evaluating model suitability and quantization trade-offs for custom GPU hardware.

---

## 🚀 Execution Pipeline & Workflow Order

Execute the pipeline scripts in the following order (either via the CLI or via the Streamlit Web UI):

```
[Option A: Web UI] streamlit run app.py
[Option B: CLI]    download_models.py  ➜  eval_real_llm.py  ➜  run_real_stats.py
```

### Step 1: Download GGUF Model Weights
Run the script below to download `Qwen2.5-7B-Instruct` model weights across 5 quantization levels (`FP16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`) from Hugging Face into the `./models/` directory:

```bash
python scripts/download_models.py
```

---

### Step 2: Run Real LLM Evaluation & Sandboxed Code Execution
Run the LLM inference evaluation on **HumanEvalPlus** problems inside the isolated sandbox:

```bash
python eval_real_llm.py
```
* **What it does**: Loads each model variant from `./models/`, generates Python code for HumanEvalPlus benchmark prompts, and validates execution against expanded fuzzed unit assertions (`base_input` + `plus_input`) inside a sandboxed environment.
* **Output**: Evaluation pass-rate matrix is saved to `./results/real_eval_matrix.json`.

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

## 📁 Project Directory Structure

```text
.
├── core/
│   ├── execution_engine.py   # Isolated code execution sandbox
│   └── recommender.py        # Hardware-aware VRAM recommendation engine
├── data/
│   └── dataset_loader.py     # HumanEvalPlus (EvalPlus) benchmark dataset loader
├── models/                   # Directory storing GGUF model weights
├── results/
│   └── real_eval_matrix.json # Generated evaluation score matrix
├── scripts/
│   └── download_models.py    # Automated Hugging Face GGUF downloader
├── stats/
│   └── stat_engine.py        # Inferential statistical engine (Friedman, Kendall, Dunn)
├── utils/
│   ├── gpu_helper.py         # Multi-vendor GPU hardware detection & AMD ROCm/HIP driver helper
│   ├── hf_helper.py          # Live Hugging Face search & GGUF file inspector
│   └── process_runner.py     # Asynchronous process runner with real-time log parsing
├── app.py                    # Interactive Streamlit Web UI Application
├── eval_real_llm.py          # Primary real LLM evaluation pipeline script
├── run_real_stats.py         # Primary statistical & recommendation execution script
├── main.py                   # End-to-end synthetic simulation prototype script
├── test_framework.py         # Framework unit test suite
├── requirements.txt          # Python package dependencies
└── README.md                 # Framework documentation
```

---

## 📜 License
This project is open-source and intended for research on LLM quantization performance trade-offs. Feel free to modify and build upon it.
