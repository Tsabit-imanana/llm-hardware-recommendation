# Penjelasan Lengkap Arsitektur Aplikasi
## Local LLM Quantization Benchmark & Hardware-Aware Recommendation Framework

> Dokumen ini adalah referensi teknis menyeluruh yang menjelaskan setiap komponen, alur data, algoritma, dan keputusan desain dari seluruh codebase.

---

## 1. Gambaran Umum Sistem

Framework ini adalah sebuah **pipeline riset end-to-end** yang dirancang untuk menjawab satu pertanyaan ilmiah:

> *"Seberapa jauh kuantisasi bobot model LLM lokal memengaruhi akurasi eksekusi kode Python, dan pada level kuantisasi mana sebuah hardware GPU tertentu masih mampu menjalankan model tanpa degradasi yang signifikan secara statistik?"*

Framework dibangun di atas **empat pilar** utama:

| Pilar | Komponen | Fungsi |
|---|---|---|
| **Akuisisi Model** | `scripts/`, `utils/hf_helper.py` | Download GGUF model dari Hugging Face |
| **Evaluasi** | `eval_real_llm.py`, `core/execution_engine.py`, `data/` | Inferensi LLM + sandboxed code execution |
| **Analisis Statistik** | `stats/stat_engine.py`, `run_real_stats.py` | Non-parametric inferential testing |
| **Rekomendasi** | `core/recommender.py` | Hardware-constrained model matching |

Seluruh pilar ini dijalankan melalui satu **Streamlit Web UI** (`app.py`) dengan background task management (`utils/process_runner.py`), atau secara langsung via CLI.

---

## 2. Struktur Direktori Lengkap

```
fpt/
│
├── app.py                         # Streamlit Web UI (661 baris)
├── eval_real_llm.py               # Pipeline evaluasi LLM nyata (132 baris)
├── run_real_stats.py              # Pipeline statistik & rekomendasi (66 baris)
├── main.py                        # Mode simulasi sintetik (163 baris)
├── test_framework.py              # Unit tests (64 baris)
├── requirements.txt               # Dependensi Python
├── README.md                      # Dokumentasi dasar
│
├── core/                          # Mesin inti eksekusi & rekomendasi
│   ├── __init__.py
│   ├── execution_engine.py        # Sandboxed code executor (103 baris)
│   └── recommender.py             # VRAM-aware model recommender (118 baris)
│
├── data/                          # Dataset HumanEval
│   ├── __init__.py
│   ├── dataset_loader.py          # Loader dataset HumanEval (31 baris)
│   └── HumanEval.jsonl.gz         # 164 problem benchmark (44KB compressed)
│
├── stats/                         # Mesin statistik inferensial
│   ├── __init__.py
│   └── stat_engine.py             # Friedman + Kendall + Dunn engine (140 baris)
│
├── utils/                         # Utilitas pendukung
│   ├── hf_helper.py               # Hugging Face API & model management (166 baris)
│   └── process_runner.py          # Async subprocess runner (167 baris)
│
├── scripts/                       # Script download standalone
│   ├── __init__.py
│   ├── download_models.py         # Preset downloader Qwen2.5-7B (100 baris)
│   └── download_repo_quants.py    # Dynamic batch downloader (86 baris)
│
├── models/                        # Direktori penyimpanan bobot GGUF
│   └── *.gguf
│
├── results/                       # Output evaluasi
│   ├── eval_config.json           # Konfigurasi run evaluasi aktif
│   └── real_eval_matrix.json      # Matriks skor pass rate hasil evaluasi
│
├── logs/                          # Execution logs
│   ├── download_config.json       # Konfigurasi download aktif
│   ├── download_execution.log     # Log real-time download progress
│   ├── real_eval_execution.log    # Log real-time LLM evaluation
│   └── real_stats_summary.log     # Log output statistik & rekomendasi
│
├── sandbox/                       # Direktori isolasi eksekusi kode sementara
│
└── vendor/                        # Dependensi lokal ter-vendor
    └── human-eval/                # Fork OpenAI HumanEval (lokal)
        └── human_eval/
            ├── data.py            # JSONL reader
            ├── execution.py       # Sandboxed execution OpenAI (referensi)
            ├── evaluation.py      # Functional correctness evaluator
            └── evaluate_functional_correctness.py
```

**Total codebase Python**: ~1.997 baris sumber utama (tidak termasuk vendor & venv).

---

## 3. Alur Kerja Pipeline (End-to-End)

```
╔══════════════════════════════════════════════════════════════╗
║                    JALUR A: Web UI                           ║
║  streamlit run app.py                                        ║
║  Browser ─► Sidebar Navigation ─► Views:                    ║
║    [1] Project Scanner                                       ║
║    [2] Model Downloader   ──► scripts/download_repo_quants.py║
║    [3] LLM Evaluation     ──► eval_real_llm.py              ║
║    [4] Statistical Summary ──► run_real_stats.py             ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    JALUR B: CLI                              ║
║  python scripts/download_models.py                           ║
║      │  (./models/*.gguf tersedia)                           ║
║  python eval_real_llm.py                                     ║
║      │  (./results/real_eval_matrix.json terbentuk)          ║
║  python run_real_stats.py                                    ║
║      │  (output terminal + ./logs/real_stats_summary.log)    ║
║  [Optional] python main.py  (synthetic simulation only)      ║
╚══════════════════════════════════════════════════════════════╝
```

### Alur Data Detail

```
Hugging Face Hub
     │ hf_hub_download()
     ▼
./models/*.gguf
     │ Llama(model_path=...)  [llama-cpp-python]
     ▼
LLM Inference Engine
     │ llm(prompt_text, max_tokens=512, temperature=0.2)
     ▼
raw_response (string)
     │ extract_python_code()
     ▼
generated_code (string)
     │ run_code_in_sandbox(generated_code, assertions)
     ▼
{ unit_test_pass_rate: float }     ← UR score per prompt
     │ accumulated per quantization level
     ▼
results_matrix = { "Q8_0": [UR1, UR2, ... URn], ... }
     │ json.dump → ./results/real_eval_matrix.json
     ▼
analyze_quantization_impact(df)
     ├─► Friedman Chi-Squared (χ²)
     ├─► Kendall's W (effect size)
     ├─► Dunn Post-Hoc (Bonferroni-adjusted pairwise p-values)
     └─► Elbow Point detection
     │
     ▼
recommend_model(stat_results, user_vram_gb)
     └─► { recommended_quantization, status, reason }
```

---

## 4. Komponen Inti: Detail Per File

### 4.1 `eval_real_llm.py` — Pipeline Evaluasi LLM Nyata

**Peran**: Skrip eksekusi utama yang menjalankan inferensi nyata menggunakan `llama-cpp-python` dan mengevaluasi kualitas output kode dengan sandboxed unit test.

**Kontrol variabel (locked / fixed):**
```python
SEED        = 42          # Reproducibility
TEMPERATURE = 0.2         # Near-deterministic generation
TOP_P       = 0.95
MAX_TOKENS  = 512
```

**`TeeLogger`** — Class kustom yang meng-intercept `sys.stdout` dan secara bersamaan menulis ke terminal dan `./logs/real_eval_execution.log`. Ini memungkinkan `process_runner.py` membaca log secara real-time dari proses anak.

**`load_eval_config()`** — Membaca `./results/eval_config.json` yang dibuat oleh UI. File ini berisi `models_map` (dict nama_quant → path file) dan `num_prompts`. Jika file tidak ada, fallback ke `DEFAULT_MODELS_MAP` (5 level Qwen2.5-7B Instruct).

**`extract_python_code(raw_response)`** — Parser markdown sederhana. Jika output LLM membungkus kode dalam blok triple-backtick, fungsi ini mengekstraknya. Jika tidak ada fence, string mentah dikembalikan.

**`run_evaluation()`** — Loop utama:
1. Untuk setiap kuantisasi dalam `models_map`:
   - Periksa apakah file `.gguf` ada. Jika tidak, lewati.
   - Load model dengan `Llama(model_path, n_ctx=2048, n_gpu_layers=-1, seed=SEED)`.
   - `n_gpu_layers=-1` berarti maksimalkan GPU offloading jika CUDA tersedia.
   - Untuk setiap HumanEval problem:
     - Buat prompt: `"Complete the following Python function. Output ONLY executable Python code inside codeblocks:\n\n{item['prompt']}"`
     - Jalankan inferensi deterministik.
     - Parse kode dari output.
     - Rekonstruksi `full_code = f"{item['prompt']}\n{gen_code}"`.
     - Evaluasi di sandbox dengan `run_code_in_sandbox(full_code, [item['test_code']])`.
     - Simpan `unit_test_pass_rate` ke `results_matrix[quant_level]`.
2. Simpan `results_matrix` ke `./results/real_eval_matrix.json`.

---

### 4.2 `core/execution_engine.py` — Sandboxed Code Executor

**Peran**: Mengevaluasi kode Python yang dihasilkan LLM secara terisolasi, per-assertion, dengan timeout protection.

**Desain utama**: Setiap assertion dieksekusi sebagai **proses terpisah** (`subprocess.run`), bukan dengan `eval()` atau `exec()` langsung. Ini memberikan isolasi nyata — infinite loop, segfault, atau import berbahaya tidak akan memengaruhi proses induk.

**Alur eksekusi per assertion:**
1. Gabungkan `generated_code + "\n\n# Unit Assertion {idx}\n{assertion}\n"` menjadi satu string skrip.
2. Tulis ke file `.py` sementara di `./sandbox/` menggunakan `tempfile.NamedTemporaryFile`.
3. Jalankan `subprocess.run([sys.executable, tmp_path], capture_output=True, timeout=timeout_sec)`.
4. Jika returncode == 0 → `s_status = 1` (passed).
5. Jika `subprocess.TimeoutExpired` → catat error `"TimeoutExpired: ..."`.
6. Hapus file sementara di blok `finally`.

**Return value:**
```python
{
    "unit_test_pass_rate": float,   # passed_count / k_total
    "results": [
        {
            "test_index": int,
            "assertion": str,
            "status": 0 | 1,       # S(C_ij, t_ik) dalam notasi paper
            "error": str | None
        }
    ],
    "total_tests": int,
    "passed_tests": int
}
```

**Metrik UR** (Unit test pass Rate): `UR(P_i, Q_j) = sum(S) / K` — rata-rata biner dari seluruh assertion.

> **Catatan penting**: Karena assertion HumanEval biasanya berupa satu fungsi `check(entry_point)` yang memanggil banyak assertion internal, `total_tests` dalam prakteknya sering bernilai 1, sehingga `unit_test_pass_rate` adalah binary (0.0 atau 1.0).

---

### 4.3 `data/dataset_loader.py` — HumanEval Loader

**Peran**: Memuat subset problem dari benchmark HumanEval (OpenAI, 2021).

**`load_humaneval_subset(limit=20)`**:
- Prioritas pertama: file lokal `./data/HumanEval.jsonl.gz`.
- Jika tidak ada, gunakan `read_problems()` dari package `human_eval`.
- Kembalikan list dict berisi: `task_id`, `prompt`, `entry_point`, `test_code`, `canonical_solution`.

**Format problem HumanEval:**
```python
{
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    ...",
    "entry_point": "has_close_elements",
    "test_code": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    ...",
    "canonical_solution": "    for idx, ...",
}
```

---

### 4.4 `stats/stat_engine.py` — Inferential Statistical Engine

**Peran**: Menjalankan seluruh pipeline statistik non-parametrik untuk menguji apakah degradasi akurasi akibat kuantisasi signifikan secara statistik.

#### Hierarki Presisi Kuantisasi

Konstanta `STANDARD_QUANT_PRECISION_ORDER` mendefinisikan urutan dari **presisi tertinggi** ke **presisi terendah**:

```python
STANDARD_QUANT_PRECISION_ORDER = [
    "F32", "FP16", "Q8_0", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_NL", "IQ4_XS",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"
]
```

Mencakup kuantisasi standar GGUF (Qx_y), extended K-quants, dan i-matrix quants (IQx). `QUANT_LEVELS_ORDER` adalah alias dari daftar ini untuk backward compatibility.

**`get_quant_order(present_levels)`**: Mengurutkan level kuantisasi yang tersedia sesuai hierarki di atas. Level yang tidak dikenal diurutkan alfabetis di akhir.

#### Pipeline `analyze_quantization_impact(matrix_data)`

Input: `dict` atau `pd.DataFrame` — kolom adalah nama kuantisasi, baris adalah skor UR per prompt.

**Langkah 1: Normalisasi urutan kolom**
```python
present_levels = get_quant_order(list(df.columns))
df = df[present_levels]
```

**Langkah 2: Friedman Test**
```python
friedman_res = stats.friedmanchisquare(*[df[q].values for q in present_levels])
```
Uji Friedman adalah alternatif non-parametrik dari ANOVA repeated measures. Cocok karena:
- Data pass rate (0.0–1.0) tidak terdistribusi normal.
- Setiap prompt diuji di semua level kuantisasi (repeated measures design).

**Langkah 3: Kendall's W (Effect Size)**
```python
kendall_w = chi2_stat / (N * (m - 1))   # dikliping ke [0.0, 1.0]
```
- W = 0: Tidak ada konsistensi antar-rater (kuantisasi).
- W = 1: Seluruh prompt setuju dalam ranking antar-level.

**Langkah 4: Dunn's Post-Hoc Test (Bonferroni)**
```python
melted_df = df.melt(var_name="quantization", value_name="pass_rate")
posthoc_df = sp.posthoc_dunn(melted_df, val_col="pass_rate",
                              group_col="quantization", p_adjust="bonferroni")
```
Menghasilkan matriks `m×m` p-value yang sudah dikoreksi Bonferroni.

**Langkah 5: Elbow Point Detection**
```python
if p_val >= 0.05:
    # Friedman tidak signifikan: tidak ada level yang benar-benar buruk
    elbow_point = present_levels[-1]  # level paling ringan aman digunakan
else:
    # Friedman signifikan: cek pairwise dari baseline
    baseline = present_levels[0]
    elbow_point = baseline
    for q_level in present_levels:
        p_adj_vs_baseline = posthoc_df.loc[baseline, q_level]
        if p_adj_vs_baseline >= 0.05:
            elbow_point = q_level   # masih tidak berbeda signifikan dari baseline
        else:
            break                   # mulai degradasi signifikan, berhenti
```

Elbow point adalah **level kuantisasi terendah yang masih tidak berbeda secara statistik dari baseline**. Ini adalah titik "aman" untuk turun presisi tanpa kehilangan akurasi bermakna.

**Return dictionary lengkap:**
```python
{
    "friedman_chi2": float,        # Chi-squared statistic
    "friedman_stat": float,        # alias friedman_chi2
    "p_value": float,              # p-value Friedman test
    "kendall_w": float,            # alias kendalls_w
    "kendalls_w": float,           # Kendall's W effect size
    "n_prompts": int,              # jumlah prompt N
    "m_levels": int,               # jumlah level kuantisasi m
    "posthoc_dunn_matrix": DataFrame,  # alias dunn_posthoc
    "dunn_posthoc": DataFrame,     # matriks p-value post-hoc
    "elbow_point": str,            # level aman terendah
    "baseline": str,               # level referensi (highest precision)
    "quant_order": list,           # urutan level yang dianalisis
    "mean_pass_rates": dict        # rata-rata UR per level
}
```

---

### 4.5 `core/recommender.py` — Hardware-Aware Recommendation Engine

**Peran**: Menentukan level kuantisasi optimal yang memenuhi dua constraint sekaligus:
1. **Hardware constraint**: `VRAM_required(quant) <= VRAM_available(user)`
2. **Statistical constraint**: `precision(quant) >= precision(elbow_point)`

**`DEFAULT_VRAM_LOOKUP_7B`** — Tabel VRAM yang dibutuhkan untuk model ~7B parameter:
```python
{
    "FP16":   14.0,   # GB
    "Q8_0":   7.5,
    "Q6_K":   6.0,
    "Q5_K_M": 5.0,
    "Q4_K_M": 4.2
}
```

**`recommend_model(stat_results, user_vram_gb, vram_lookup=None)`**:

1. **Ambil quant_order** dari `stat_results["quant_order"]` yang sudah disortir oleh `analyze_quantization_impact`.
2. **Bangun `lookup_map`** (VRAM requirements):
   - Isi dengan `DEFAULT_VRAM_LOOKUP_7B` untuk level yang dikenal.
   - Untuk level yang tidak ada di tabel default, ambil estimasi dari **ukuran file model nyata** di `./models/`: `size_gb + 1.0 GB overhead`.
   - Fallback default: 5.0 GB jika tidak diketahui.
3. **Filter `valid_precision_candidates`** — semua level dengan `index <= elbow_idx` (presisi >= elbow point).
4. **Filter `feasible_candidates`** — semua level yang `VRAM_req <= user_vram_gb`.
5. **Pilih best**: `min(feasible_candidates, key=lambda q: lookup_map[q])` — VRAM paling kecil yang masih valid.
6. Jika kosong → `status = "INSUFFICIENT_VRAM"`.

---

### 4.6 `run_real_stats.py` — CLI Statistical Runner

**Peran**: Skrip CLI yang membaca `real_eval_matrix.json`, menjalankan `analyze_quantization_impact`, mencetak hasil ke terminal, dan menyimpannya ke `./logs/real_stats_summary.log` melalui `TeeLogger`.

**Output terminal mencakup:**
- Tabel deskriptif (`df.describe()`).
- Friedman χ², p-value, Kendall's W, Elbow Point.
- Matriks post-hoc Dunn's (full pairwise).
- Hardware Recommendation Matrix untuk VRAM: 16, 8, 6, 5, 4 GB.

---

### 4.7 `main.py` — Synthetic Simulation Mode

**Peran**: Menjalankan seluruh pipeline menggunakan data sintetik (tanpa model nyata). Berguna untuk validasi pipeline statistik secara cepat.

**`generate_synthetic_evaluation_matrix(n_prompts=20, seed=42)`**:
```python
base_ability = np.random.uniform(0.70, 0.98, size=n_prompts)
matrix = {
    "FP16":   clip(base_ability + N(0, 0.01)),
    "Q8_0":   clip(base_ability - 0.01 + N(0, 0.015)),
    "Q6_K":   clip(base_ability - 0.02 + N(0, 0.02)),
    "Q5_K_M": clip(base_ability - 0.08 + N(0, 0.03)),
    "Q4_K_M": clip(base_ability - 0.18 + N(0, 0.04)),
}
```
Memodelkan degradasi kuantisasi secara realistis: semakin rendah bit-width, semakin besar deviasi negatif dari performa ideal. seed=42 untuk reproducibility.

---

### 4.8 `utils/hf_helper.py` — Hugging Face Integration

**Peran**: Semua interaksi dengan Hugging Face Hub — pencarian, listing file, dan download model.

**`DownloadProgressLogger(tqdm)`**: Subclass dari `tqdm` yang meng-override `update()` untuk mencetak progress dengan format khusus yang dapat di-parse regex:
```
PROGRESS: [filename.gguf] 45% (1980.0 MB / 4400.0 MB)
```

**Fungsi-fungsi:**

| Fungsi | Output | Keterangan |
|---|---|---|
| `search_hf_models(query, limit)` | `list[dict]` | Cari repo via `HfApi.list_models()` |
| `list_repo_gguf_files(repo_id)` | `list[str]` | Daftar nama file `.gguf` |
| `list_repo_gguf_files_info(repo_id)` | `list[dict]` | Dengan ukuran file via `model_info(files_metadata=True)` |
| `get_local_models()` | `list[dict]` | Walk `./models/`, metadata setiap `.gguf`/`.bin` |
| `download_hf_file(repo_id, filename, ...)` | `(bool, path, msg)` | Download via `hf_hub_download` + progress callback |

`get_local_models()` mengembalikan dict per file dengan key: `filename`, `rel_path`, `abs_path`, `size_mb`, `size_gb`, `mod_time`.

---

### 4.9 `utils/process_runner.py` — Background Task Runner

**Peran**: Menjalankan skrip Python eksternal sebagai proses anak non-blocking, streaming stdout-nya ke memori line-by-line, dan meng-parse progress dari log output.

#### Class `BackgroundTaskRunner`

**State internal:**
```python
self.process = None           # subprocess.Popen handle
self.log_lines = []           # seluruh baris stdout yang sudah dibaca
self.is_running = False       # apakah proses sedang berjalan
self.exit_code = None         # kode keluar proses
self.lock = threading.Lock()  # thread safety
self.progress_percentage = float  # 0.0 - 1.0
self.current_status = str     # status teks saat ini
self.current_quant = str      # level kuantisasi yang sedang diproses
self.current_prompt_index = int
self.total_prompts = int
```

**`start_task(command_args, cwd=None)`**:
- Guard: tolak jika sudah ada proses berjalan.
- Set env `PYTHONUNBUFFERED=1` agar output tidak di-buffer.
- Buat `subprocess.Popen` dengan `stdout=PIPE, stderr=STDOUT` (stderr digabung).
- Start daemon thread `_read_output` untuk membaca output asynchronous.

**`_read_output()`** (thread daemon):
- Loop `iter(process.stdout.readline, '')` — blocking, baca satu baris.
- Untuk setiap baris: tambahkan ke `log_lines`, panggil `_parse_line()`.
- Setelah proses selesai: update `exit_code`, set `is_running=False`.

**`_parse_line(line)` — Regex Progress Parser:**

Parser adalah jembatan antara teks log dan progress bar UI. Empat pola:

```python
# Pattern 1: Header evaluasi kuantisasi
r"Running Evaluation for Quantization:\s*(.+)$"

# Pattern 2: Progress per prompt
r"\[(.+?)\]\s+Prompt\s+(\d+)/(\d+)"
# Baca eval_config.json untuk total model yang dijalankan

# Pattern 3: Progress download
r"PROGRESS:\s*\[(.*?)\]\s*(\d+)%\s*\((.*?)\)"

# Pattern 4: Batch download item
r"\[(\d+)/(\d+)\]\s+STARTING DOWNLOAD:\s*(.*)"
```

**Tiga instance global:**
```python
eval_runner     = BackgroundTaskRunner()   # untuk eval_real_llm.py
stats_runner    = BackgroundTaskRunner()   # untuk run_real_stats.py
download_runner = BackgroundTaskRunner()   # untuk download_repo_quants.py
```

---

### 4.10 `scripts/download_models.py` — Preset Model Downloader

**Peran**: Script CLI untuk mengunduh 5 level kuantisasi Qwen2.5-7B-Instruct dari dua source alternatif.

**`QUANT_CONFIGS`**: Dict `{quant_level: [(repo_id, remote_spec), ...]}`. `remote_spec` bisa string (file tunggal) atau list (sharded files).

**Logika fallback**: Jika download dari repo pertama gagal, coba repo berikutnya. Untuk file sharded, semua shard diunduh lalu shard pertama dicopy sebagai entry point.

---

### 4.11 `scripts/download_repo_quants.py` — Dynamic Batch Downloader

**Peran**: Downloader generik yang menerima konfigurasi via argumen CLI atau file JSON. Digunakan oleh Web UI saat user men-download dari hasil pencarian Hugging Face.

**Argument parsing:**
```bash
python scripts/download_repo_quants.py --config ./logs/download_config.json
# atau
python scripts/download_repo_quants.py --repo org/repo-GGUF file1.gguf file2.gguf
```

**`download_config.json`** (dibuat oleh UI):
```json
{
    "repo_id": "unsloth/Qwen3.5-4B-GGUF",
    "files": ["Qwen3.5-4B-IQ4_XS.gguf", "Qwen3.5-4B-UD-IQ2_XXS.gguf"]
}
```

---

### 4.12 `app.py` — Streamlit Web UI Application

**Peran**: Antarmuka web utama yang mengintegrasikan seluruh komponen. 661 baris dengan dark glassmorphism theme.

#### Inisialisasi

```python
# Singleton BackgroundTaskRunner dari process_runner
from utils.process_runner import eval_runner, stats_runner, download_runner
```

Karena Streamlit berjalan di satu proses Python, objek runner bersifat **global module-level**. State mereka bertahan antar-refresh page.

#### Sidebar Navigation

4 view dikontrol oleh `st.sidebar.radio`. GPU detection dilakukan dengan `torch.cuda` jika tersedia.

#### View 1: Project Overview (`view_project_scanner`)

- **`scan_project()`**: Walk semua `.py` di project root, hitung baris dan ukuran.
- Metric cards: jumlah file Python, total model size, status matriks, ukuran log.
- DataFrame sortable: file Python, GGUF models, ringkasan modul.

#### View 2: Model Downloader (`view_model_downloader`)

**Tab 1 — Preset Downloads (Qwen2.5-7B)**:
- Grid 5 kolom: FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M.
- Auto-deteksi file yang sudah ada di `./models/`.
- Tombol download → `download_models.py` via `download_runner`.

**Tab 2 — Live Hugging Face Search**:
- Search box → `search_hf_models()` → dropdown repo.
- Pilih repo → `list_repo_gguf_files_info()` → multiselect file GGUF.
- Tombol download → tulis `download_config.json` → `download_repo_quants.py`.

**Tab 3 — Downloaded Models Manager**:
- List model dengan ukuran dan timestamp.
- Tombol Delete per file.

**Live Progress Banner**: progress bar + log terminal 25 baris terakhir. Auto-rerun 1 detik selama running.

#### View 3: LLM Evaluation (`view_eval_runner`)

**`get_quant_label(filename)`**: Ekstrak label bersih dari nama file:
- `Qwen3.5-4B-IQ4_XS.gguf` → `IQ4_XS`
- `qwen2.5-7b-instruct-q4_k_m.gguf` → `Q4_K_M`

Didukung: semua pola standar GGUF + IQ quants.

**Saat Start diklik**:
1. Build `models_map_preview = {label: path}`.
2. Tulis ke `./results/eval_config.json`.
3. Jalankan `eval_real_llm.py` via `eval_runner.start_task()`.

**Progress Visualization**:
- Progress bar 0-100%.
- Caption: current quant, prompt index, total progress.
- Log terminal 30 baris.
- Live charts (bar chart rata-rata UR, line chart per-prompt).

#### View 4: Statistical Summary (`view_stat_summary`)

- Tombol "Run Statistical Pipeline" → `run_real_stats.py` via `stats_runner`.
- Baca `real_eval_matrix.json` → `analyze_quantization_impact()` langsung (bukan subprocess, untuk interaktivitas).
- 4 metric cards: Friedman χ², p-value, Kendall's W, Elbow Point.
- Heatmap Dunn post-hoc (Plotly, dark theme, Viridis colorscale).
- Slider VRAM (1–32 GB) → `recommend_model()` real-time → card rekomendasi.
- Tabel hardware matrix: 6 threshold VRAM.

---

### 4.13 `test_framework.py` — Unit Tests

Suite `unittest` untuk memvalidasi komponen inti tanpa memerlukan model nyata.

| Test | Yang Diuji |
|---|---|
| `test_execution_engine_success` | Kode valid → pass rate 1.0 |
| `test_execution_engine_timeout` | Infinite loop → TimeoutExpired |
| `test_execution_engine_syntax_error` | Syntax error → caught |
| `test_statistical_engine` | Output dict berisi semua key yang diharapkan |
| `test_recommendation_engine_optimal` | Elbow Q5_K_M + 8GB → OPTIMAL, Q5_K_M |
| `test_recommendation_engine_insufficient_vram` | Elbow Q5_K_M + 4GB → INSUFFICIENT_VRAM |

---

## 5. Vendor: OpenAI HumanEval

`vendor/human-eval/` adalah fork lokal dari [openai/human-eval](https://github.com/openai/human-eval) yang di-install sebagai package editable.

**`human_eval/data.py`**: `read_problems(evalset_file)` membaca `.jsonl.gz`, yield setiap baris sebagai dict.

**`human_eval/execution.py`**: Evaluator asli OpenAI menggunakan `multiprocessing.Process` + `reliability_guard()` (neutralisasi fungsi OS berbahaya seperti `os.kill`, `subprocess.Popen`, dll). **Tidak digunakan** dalam pipeline utama — framework menggunakan `core/execution_engine.py` sendiri. File ini ada sebagai referensi/alternatif.

**`data/HumanEval.jsonl.gz`**: 164 problem Python coding dengan prompt, test assertions, dan solusi kanonik.

---

## 6. Runtime Files (Artifacts)

### `results/eval_config.json`
Dibuat oleh UI sebelum menjalankan evaluasi:
```json
{
    "models_map": {
        "IQ4_XS": "./models/Qwen3.5-4B-IQ4_XS.gguf",
        "IQ2_XXS": "./models/Qwen3.5-4B-UD-IQ2_XXS.gguf",
        "IQ3_XXS": "./models/Qwen3.5-4B-UD-IQ3_XXS.gguf"
    },
    "num_prompts": 20
}
```

### `results/real_eval_matrix.json`
Output dari `eval_real_llm.py`:
```json
{
    "IQ4_XS":  [1.0, 0.0, 1.0, ...],
    "IQ2_XXS": [1.0, 1.0, 1.0, ...],
    "IQ3_XXS": [0.0, 0.0, 1.0, ...]
}
```

### Log Files

| File | Ditulis oleh | Isi |
|---|---|---|
| `real_eval_execution.log` | `eval_real_llm.py` via TeeLogger | Stdout evaluasi termasuk llama.cpp verbose |
| `real_stats_summary.log` | `run_real_stats.py` via TeeLogger | Hasil statistik & tabel rekomendasi |
| `download_execution.log` | `download_repo_quants.py` via TeeLogger | Progress download per file |

---

## 7. Dependency Graph (Import)

```
app.py
  ├── utils.hf_helper       (search, list, download, get_local)
  ├── utils.process_runner  (eval_runner, stats_runner, download_runner)
  ├── stats.stat_engine     (analyze_quantization_impact)
  └── core.recommender      (recommend_model, DEFAULT_VRAM_LOOKUP_7B)

eval_real_llm.py
  ├── llama_cpp.Llama
  ├── data.dataset_loader   (load_humaneval_subset)
  └── core.execution_engine (run_code_in_sandbox)

run_real_stats.py
  ├── stats.stat_engine     (analyze_quantization_impact)
  └── core.recommender      (recommend_model)

main.py
  ├── core.execution_engine (run_code_in_sandbox)
  ├── stats.stat_engine     (analyze_quantization_impact)
  └── core.recommender      (recommend_model)

core.recommender
  ├── stats.stat_engine     (STANDARD_QUANT_PRECISION_ORDER, get_quant_order)
  └── utils.hf_helper       [runtime import: get_local_models, get_quant_label]

data.dataset_loader
  └── human_eval.data       (read_problems)
```

---

## 8. Dependensi Python

```
numpy>=1.24.0          # Operasi array, random generation
pandas>=2.0.0          # DataFrame untuk matriks evaluasi
scipy>=1.10.0          # stats.friedmanchisquare
scikit-posthocs>=0.7.0 # sp.posthoc_dunn (Dunn's test)
tabulate>=0.9.0        # Tabel terminal (main.py, run_real_stats.py)
huggingface_hub        # HfApi, hf_hub_download
llama-cpp-python       # Llama class untuk inferensi GGUF lokal
streamlit>=1.30.0      # Web UI framework
plotly>=5.18.0         # Visualisasi interaktif (bar, line, heatmap)
```

**Dependensi vendor:**
```
human-eval @ git+https://github.com/openai/human-eval.git
```

**Opsional (runtime-detected):**
```
torch  # Untuk GPU detection di sidebar
```

---

## 9. Thread Safety & Concurrency Model

Streamlit menjalankan semua interaksi user di **satu thread Python**, namun me-rerun seluruh script setiap ada interaksi UI.

`BackgroundTaskRunner` menggunakan:
- `threading.Lock` untuk melindungi semua akses ke state (`log_lines`, `is_running`, dll).
- Daemon thread terpisah untuk membaca stdout proses anak.
- `get_state()` selalu mengambil lock — aman dipanggil dari Streamlit main thread.

Pola auto-rerun:
```python
if is_running:
    time.sleep(1)
    st.rerun()
```
Polling loop 1 detik yang memperbarui UI selama proses berjalan.

---

## 10. Pola Desain & Keputusan Arsitektur

### Mengapa subprocess, bukan threading untuk skrip evaluasi?
`eval_real_llm.py` dan `run_real_stats.py` adalah skrip mandiri yang menggunakan `TeeLogger` untuk meng-intercept `sys.stdout`. Jika dijalankan sebagai thread dalam proses Streamlit, stdout akan bertabrakan antar-thread. Subprocess memberi isolasi penuh.

### Mengapa TeeLogger, bukan file logging biasa?
`process_runner.py` membaca stdout dari proses anak via `subprocess.PIPE`. Jika skrip hanya menulis ke file log (tidak ke stdout), runner tidak bisa membacanya. TeeLogger memastikan output tersedia di kedua tempat secara bersamaan.

### Mengapa `PYTHONUNBUFFERED=1`?
Tanpa ini, Python mem-buffer output ke stdout. Pada proses yang berjalan berjam-jam, buffer tidak akan di-flush ke pipe sampai penuh atau proses selesai. Dengan `PYTHONUNBUFFERED`, setiap `print(..., flush=True)` langsung tersedia di pipe dan bisa dibaca oleh `process_runner.py`.

### Mengapa format `PROGRESS: [filename] X% (MB/MB)`?
Format ini dipilih karena: (1) mudah di-parse dengan satu regex, (2) tidak ambiguous dengan output llama.cpp yang verbose, (3) mengandung semua informasi yang dibutuhkan UI dalam satu baris.

### Mengapa `STANDARD_QUANT_PRECISION_ORDER` sebagai konstanta global di `stat_engine`?
Agar urutan hierarki presisi konsisten di seluruh sistem. `stat_engine.py` (sorting kolom DataFrame), `recommender.py` (sorting kandidat), dan `app.py` (label extraction) semua mengacu ke sumber yang sama.

### Mengapa elbow point menggunakan iterasi greedy?
Elbow point didefinisikan sebagai **transisi pertama** dari "tidak berbeda signifikan" ke "berbeda signifikan". Iterasi greedy berhenti tepat di transisi pertama. Threshold global (ambil semua dengan p >= 0.05) tidak tepat karena bisa melewati transisi.

### Mengapa sandbox menggunakan subprocess per-assertion, bukan exec()?
Memberikan isolasi nyata: infinite loop, segfault, atau kode berbahaya tidak mempengaruhi proses induk. Setiap assertion mendapat proses Python baru dengan timeout independent.

---

## 11. Model yang Saat Ini Tersimpan

```
./models/
├── Qwen3.5-4B-IQ4_XS.gguf              (2.31 GB)  - i-matrix 4-bit extra small
├── Qwen3.5-4B-UD-IQ2_XXS.gguf          (1.42 GB)  - i-matrix 2-bit double extra small
├── Qwen3.5-4B-UD-IQ3_XXS.gguf          (1.82 GB)  - i-matrix 3-bit double extra small
├── qwen2.5-7b-instruct-fp16.gguf        (15.24 GB) - full precision 16-bit
├── qwen2.5-7b-instruct-q8_0.gguf        (8.10 GB)  - 8-bit quantized
├── qwen2.5-7b-instruct-q6_k.gguf        (6.25 GB)  - 6-bit K-quant
├── qwen2.5-7b-instruct-q5_k_m.gguf      (5.44 GB)  - 5-bit K-quant medium
├── qwen2.5-7b-instruct-q4_k_m.gguf      (4.68 GB)  - 4-bit K-quant medium
└── qwen2.5-7b-instruct-fp16-0000[1-4]-of-00004.gguf  (shard files FP16)
```

Total disk usage: ~45+ GB.

---

## 12. Cara Menambahkan Model atau Dataset Baru

### Menambah Model Baru (Custom)
1. Download file `.gguf` ke `./models/` (bisa via Tab 2 di Web UI, atau manual).
2. Model muncul otomatis di multiselect pada Evaluation View.
3. `get_quant_label()` mengenali quant type dari nama file. Jika dikenal (`iq4_xs`, `q4_k_m`, dll), label otomatis dihasilkan. Jika tidak, nama file (tanpa ekstensi) digunakan.

### Menambah Level Kuantisasi Baru ke Hierarki
Edit `STANDARD_QUANT_PRECISION_ORDER` di `stats/stat_engine.py` — tambahkan pada posisi sesuai dalam urutan presisi.

### Menggunakan Dataset Evaluasi Berbeda
Ganti `data/HumanEval.jsonl.gz` dengan dataset `.jsonl.gz` lain, dan sesuaikan `data/dataset_loader.py` agar mengembalikan format `{task_id, prompt, test_code}` yang sama.

---

## 13. Menjalankan Unit Tests

```bash
cd /home/tsabit/Python/fpt
./venv/bin/python -m pytest test_framework.py -v
# atau tanpa pytest:
./venv/bin/python test_framework.py
```

Semua 6 test case harus lulus tanpa memerlukan model GGUF (menggunakan synthetic data dan kode Python sederhana).

---

*Dokumen ini di-generate secara menyeluruh dari scan seluruh codebase — 2026-08-07*
