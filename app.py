import os
import sys
import json
import time
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hf_helper import (
    search_hf_models,
    list_repo_gguf_files,
    list_repo_gguf_files_info,
    get_local_models,
    download_hf_file
)
from utils.process_runner import eval_runner, stats_runner, download_runner
from stats.stat_engine import analyze_quantization_impact
from core.recommender import recommend_model, DEFAULT_VRAM_LOOKUP_7B

# Page Configuration
st.set_page_config(
    page_title="LLM Hardware Recommendation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism aesthetic)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F9FAFB;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .status-badge-running {
        background-color: #059669;
        color: #ECFDF5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-idle {
        background-color: #4B5563;
        color: #F3F4F6;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .log-terminal {
        background-color: #0D1117;
        color: #38BDF8;
        font-family: 'Courier New', Courier, monospace;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1E293B;
        height: 320px;
        overflow-y: auto;
        font-size: 0.85rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


def scan_project():
    """Scans project directory structure and metadata."""
    py_files = []
    total_lines = 0
    for root, _, files in os.walk(PROJECT_ROOT):
        if ".git" in root or "__pycache__" in root or "venv" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                rel_p = os.path.relpath(path, PROJECT_ROOT)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file:
                        lines = len(file.readlines())
                        total_lines += lines
                        py_files.append({"File": rel_p, "Lines": lines, "Size (KB)": round(os.path.getsize(path)/1024, 2)})
                except Exception:
                    pass
    return py_files, total_lines


def render_sidebar():
    st.sidebar.markdown("## ⚡ Navigation")
    view = st.sidebar.radio(
        "Select Workflow View:",
        [
            "🔍 Project Overview & Scanner",
            "📥 Model Downloader (Hugging Face)",
            "🖥️ GPU & Hardware Config (AMD / NVIDIA)",
            "⚡ LLM Evaluation (eval_real_llm)",
            "📊 Statistical Summary (run_real_stat)"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Hardware Context")
    
    try:
        from utils.gpu_helper import detect_gpu_hardware
        hw = detect_gpu_hardware()
        if hw["vendor"] != "CPU":
            st.sidebar.success(f"GPU Detected:\n**{hw['vendor']}** - {hw['gpu_name'][:25]}\nVRAM: {hw['vram_free_gb']} GB Free / {hw['vram_total_gb']} GB Total\nBackend: **{hw['recommended_backend']}**")
        else:
            st.sidebar.warning(f"CPU Mode / Fallback\nRAM Free: {hw['vram_free_gb']} GB / {hw['vram_total_gb']} GB")
    except Exception:
        st.sidebar.info("Hardware Detection Unavailable")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Local LLM Recommendation Framework**\nReal-time evaluation, quantization degradation stats, and hardware matching.")

    return view


def view_project_scanner():
    st.markdown('<div class="main-header">🔍 Project Overview & Workspace Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated discovery of codebase structure, model artifacts, evaluation matrices, and execution logs.</div>', unsafe_allow_html=True)

    py_files, total_lines = scan_project()
    local_models = get_local_models()
    
    matrix_exists = os.path.exists("./results/real_eval_matrix.json")
    matrix_info = "Available" if matrix_exists else "Missing"
    
    eval_log_exists = os.path.exists("./logs/real_eval_execution.log")
    eval_log_size = f"{os.path.getsize('./logs/real_eval_execution.log')/1024:.1f} KB" if eval_log_exists else "0 KB"

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Python Files", len(py_files), delta=f"{total_lines} total lines")
    with col2:
        st.metric("Downloaded Models", len(local_models), delta=f"{sum(m['size_gb'] for m in local_models):.2f} GB Total")
    with col3:
        st.metric("Eval Matrix Status", matrix_info, delta="results/real_eval_matrix.json")
    with col4:
        st.metric("Execution Log Size", eval_log_size, delta="logs/real_eval_execution.log")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📁 Python Codebase Structure")
        df_files = pd.DataFrame(py_files)
        st.dataframe(df_files, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("💾 Cached GGUF Models (./models)")
        if local_models:
            df_models = pd.DataFrame(local_models)[["filename", "size_gb", "mod_time"]]
            df_models.columns = ["Model File", "Size (GB)", "Last Modified"]
            st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.info("No GGUF model files found in `./models`. Go to the Model Downloader page to download models.")

    st.markdown("---")
    st.subheader("📜 Core Project Scripts & Modules")
    script_summary = [
        {"Script / Module": "eval_real_llm.py", "Purpose": "Evaluates llama-cpp GGUF quantizations on HumanEvalPlus (EvalPlus) benchmark prompts.", "Status": "Ready"},
        {"Script / Module": "run_real_stats.py", "Purpose": "Computes Friedman test, Kendall's W, Dunn post-hoc matrix, and recommendations.", "Status": "Ready"},
        {"Script / Module": "scripts/download_models.py", "Purpose": "Downloads default Qwen2.5 GGUF quantizations from Hugging Face.", "Status": "Ready"},
        {"Script / Module": "core/execution_engine.py", "Purpose": "Sandboxed Python code execution with assertion verification.", "Status": "Ready"},
        {"Script / Module": "stats/stat_engine.py", "Purpose": "Non-parametric inferential statistical pipeline (Friedman + Dunn).", "Status": "Ready"},
        {"Script / Module": "core/recommender.py", "Purpose": "Hardware VRAM threshold matching algorithm.", "Status": "Ready"},
    ]
    st.dataframe(pd.DataFrame(script_summary), use_container_width=True, hide_index=True)


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



def view_model_downloader():
    st.markdown('<div class="main-header">📥 Hugging Face Model Downloader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search model repositories on Hugging Face, preview all available quantized model variants, and batch download.</div>', unsafe_allow_html=True)

    dl_state = download_runner.get_state()
    
    # Download Progress Banner & Live Terminal
    if dl_state["is_running"]:
        p_val = dl_state["progress"]
        pct_int = int(p_val * 100)
        st.info(f"⏳ **Active Download Process**: {dl_state['status']}")
        st.progress(p_val)
        st.caption(f"**Download Progress**: `{pct_int}%` | **Status**: {dl_state['status']}")
        
        if st.button("🚫 Cancel Active Download"):
            download_runner.stop_task()
            st.rerun()
            
        st.subheader("📜 Live Download Output Terminal")
        logs_text = "\n".join(dl_state["logs"][-25:]) if dl_state["logs"] else "Initializing download process..."
        st.markdown(f'<div class="log-terminal">{logs_text}</div>', unsafe_allow_html=True)
        st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔥 Preset GGUF Quantizations", "🌐 Live Hugging Face Search", "💾 Downloaded Models Manager"])

    with tab1:
        st.subheader("Default Project Model Preset (Qwen2.5-7B-Instruct GGUF)")
        st.write("Download predefined quantization levels for the evaluation benchmark suite:")

        presets = [
            {"Quant": "FP16", "Filename": "qwen2.5-7b-instruct-fp16.gguf", "Est Size": "~15.2 GB", "Source": "bartowski / Qwen"},
            {"Quant": "Q8_0", "Filename": "qwen2.5-7b-instruct-q8_0.gguf", "Est Size": "~7.7 GB", "Source": "bartowski / Qwen"},
            {"Quant": "Q6_K", "Filename": "qwen2.5-7b-instruct-q6_k.gguf", "Est Size": "~5.9 GB", "Source": "bartowski / Qwen"},
            {"Quant": "Q5_K_M", "Filename": "qwen2.5-7b-instruct-q5_k_m.gguf", "Est Size": "~5.2 GB", "Source": "bartowski / Qwen"},
            {"Quant": "Q4_K_M", "Filename": "qwen2.5-7b-instruct-q4_k_m.gguf", "Est Size": "~4.4 GB", "Source": "bartowski / Qwen"}
        ]
        
        cols = st.columns(len(presets))
        local_files = [m["filename"] for m in get_local_models()]

        for idx, p in enumerate(presets):
            with cols[idx]:
                st.markdown(f"#### {p['Quant']}")
                st.caption(f"**Size**: {p['Est Size']}\n\n**File**: `{p['Filename']}`")
                
                is_downloaded = p["Filename"] in local_files
                if is_downloaded:
                    st.success("✓ Downloaded")
                else:
                    if st.button(f"Download {p['Quant']}", key=f"dl_preset_{p['Quant']}"):
                        cmd = [sys.executable, "scripts/download_models.py"]
                        download_runner.start_task(cmd, cwd=PROJECT_ROOT)
                        st.rerun()

    with tab2:
        st.subheader("Search Hugging Face Model Repositories")
        
        search_query = st.text_input("🔍 Search Model Repositories (e.g. `Qwen2.5-7B`, `Llama-3`, `GGUF`)", value="GGUF")
        
        if st.button("🔍 Search Hugging Face Hub"):
            with st.spinner("Searching Hugging Face API..."):
                st.session_state["hf_search_results"] = search_hf_models(search_query, limit=12)
        
        if "hf_search_results" not in st.session_state:
            st.session_state["hf_search_results"] = search_hf_models("GGUF", limit=12)
        
        hf_results = st.session_state["hf_search_results"]
        
        if hf_results:
            repo_options = [r["id"] for r in hf_results]
            selected_repo = st.selectbox("Select Model Repository:", repo_options)
            
            if selected_repo:
                st.caption(f"Inspecting repository `.gguf` quantized model variants for: `{selected_repo}`")
                
                with st.spinner(f"Fetching quantized files list for {selected_repo}..."):
                    gguf_info_list = list_repo_gguf_files_info(selected_repo)
                
                if gguf_info_list:
                    all_filenames = [g["filename"] for g in gguf_info_list]
                    
                    st.markdown("### 📋 Quantization Models Confirmation (Pre-Download List)")
                    st.info("The repository contains the following quantized model files. **All detected quantization models are selected by default** so you can automatically download the complete model suite for evaluation.")
                    
                    # Selection Form / Multi-select
                    selected_files = st.multiselect(
                        "Confirm / Select Quantized Model Files to Download:",
                        options=all_filenames,
                        default=all_filenames,
                        key="selected_quants_multiselect"
                    )
                    
                    # Display breakdown table
                    df_preview = pd.DataFrame([g for g in gguf_info_list if g["filename"] in selected_files])
                    if not df_preview.empty:
                        st.dataframe(df_preview[["filename", "size_gb"]].rename(columns={"filename": "Quantized Model File", "size_gb": "Est. Size (GB)"}), use_container_width=True, hide_index=True)
                    
                    if selected_files:
                        if st.button(f"🚀 Confirm & Download All Selected Quantized Models ({len(selected_files)} Files)", type="primary", use_container_width=True):
                            cfg_path = os.path.join(PROJECT_ROOT, "logs", "download_config.json")
                            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
                            with open(cfg_path, "w", encoding="utf-8") as f:
                                json.dump({"repo_id": selected_repo, "files": selected_files}, f, indent=2)
                            
                            cmd = [sys.executable, "scripts/download_repo_quants.py", "--config", cfg_path]
                            download_runner.start_task(cmd, cwd=PROJECT_ROOT)
                            st.rerun()
                    else:
                        st.warning("Please select at least one quantized model file to download.")
                else:
                    st.warning(f"No `.gguf` files found directly in root of `{selected_repo}`. Try another GGUF repo like `bartowski/Qwen2.5-7B-Instruct-GGUF`.")

    with tab3:
        st.subheader("Manage Saved Models in `./models/`")
        models = get_local_models()
        if models:
            for m in models:
                col_m1, col_m2, col_m3 = st.columns([3, 1, 1])
                with col_m1:
                    st.write(f"📄 **{m['filename']}** ({m['size_gb']:.2f} GB)")
                    st.caption(f"Path: `{m['rel_path']}` | Modified: {m['mod_time']}")
                with col_m2:
                    st.info("Verified")
                with col_m3:
                    if st.button("🗑️ Delete", key=f"del_{m['filename']}"):
                        try:
                            os.remove(m['abs_path'])
                            st.success(f"Deleted {m['filename']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                st.markdown("---")
        else:
            st.info("No model files in `./models/` directory.")

    if dl_state["is_running"]:
        time.sleep(1)
        st.rerun()


def view_eval_runner():
    st.markdown('<div class="main-header">⚡ Real LLM Evaluation Engine (eval_real_llm)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select target models, configure prompt sample size, and run HumanEvalPlus (EvalPlus) code generation evaluation.</div>', unsafe_allow_html=True)

    state = eval_runner.get_state()
    is_running = state["is_running"]

    local_models = get_local_models()
    
    st.markdown("### 🎯 Choose Models & Prompts for Evaluation")
    
    if local_models:
        all_model_filenames = [m["filename"] for m in local_models]
        
        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            selected_model_files = st.multiselect(
                "Choose Models to Evaluate:",
                options=all_model_filenames,
                default=all_model_filenames,
                help="Select which downloaded GGUF models in ./models/ to benchmark."
            )
        with col_cfg2:
            num_prompts = st.slider(
                "Prompt Sample Size (N):",
                min_value=1,
                max_value=20,
                value=20,
                help="Number of HumanEvalPlus (EvalPlus) benchmark prompts to evaluate per model variant."
            )
        
        # Build mapping & display table
        models_map_preview = {}
        for fname in selected_model_files:
            m_path = f"./models/{fname}"
            label = get_quant_label(fname)
            # Ensure unique keys if duplicate labels exist
            if label in models_map_preview:
                label = f"{label}_{fname[:6]}"
            models_map_preview[label] = m_path

        if selected_model_files:
            with st.expander("🔍 View Selected Evaluation Mapping", expanded=False):
                df_map = pd.DataFrame([
                    {"Quantization Label": k, "Model Path": v, "File Status": "Exists" if os.path.exists(v) else "Missing"}
                    for k, v in models_map_preview.items()
                ])
                st.dataframe(df_map, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Please select at least one model file to evaluate.")
    else:
        st.warning("⚠️ No downloaded `.gguf` model files found in `./models/`. Go to the Model Downloader tab to download models first.")
        selected_model_files = []
        models_map_preview = {}
        num_prompts = 20

    st.markdown("---")

    # Action Toolbar
    col_btn1, col_btn2, col_status = st.columns([2, 1.5, 5])
    with col_btn1:
        if not is_running:
            btn_disabled = len(selected_model_files) == 0
            if st.button("▶️ Start Evaluation with Selected Models", type="primary", use_container_width=True, disabled=btn_disabled):
                # Save eval_config.json
                cfg_path = os.path.join(PROJECT_ROOT, "results", "eval_config.json")
                os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "models_map": models_map_preview,
                        "num_prompts": num_prompts
                    }, f, indent=2)
                
                cmd = [sys.executable, "eval_real_llm.py"]
                eval_runner.start_task(cmd, cwd=PROJECT_ROOT)
                st.rerun()
        else:
            st.button("▶️ Evaluation Running...", disabled=True, use_container_width=True)
    with col_btn2:
        if is_running:
            if st.button("⏹️ Stop Execution", type="secondary", use_container_width=True):
                eval_runner.stop_task()
                st.rerun()

    with col_status:
        if is_running:
            st.markdown('<span class="status-badge-running">● RUNNING</span> &nbsp; ' + f"**{state['status']}**", unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-idle">● IDLE</span> &nbsp; ' + f"Status: {state['status']}", unsafe_allow_html=True)

    st.markdown("---")

    # Real-Time Progress Bar Section
    st.subheader("📈 Real-Time Progress Visualization")
    progress_val = state["progress"]
    st.progress(progress_val)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.caption(f"**Current Quantization Level**: `{state['quant'] or 'N/A'}`")
    with col_p2:
        st.caption(f"**Evaluated Prompt**: `{state['prompt_idx']}/{state['total_prompts']}`")
    with col_p3:
        st.caption(f"**Total Progress**: `{int(progress_val * 100)}%`")

    # Real-time stdout log terminal
    st.subheader("📜 Live Output Console Logs")
    logs_text = "\n".join(state["logs"][-30:]) if state["logs"] else "No evaluation log output yet. Click 'Start Evaluation' above."
    st.markdown(f'<div class="log-terminal">{logs_text}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Live Evaluation Results Matrix")

    # Load matrix if file exists
    matrix_file = "./results/real_eval_matrix.json"
    if os.path.exists(matrix_file):
        try:
            with open(matrix_file, "r") as f:
                data = json.load(f)
            df_matrix = pd.DataFrame(data)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### Mean Unit Test Pass Rate (UR) per Quantization")
                mean_series = df_matrix.mean().reset_index()
                mean_series.columns = ["Quantization Level", "Mean UR Pass Rate"]
                
                fig_bar = px.bar(
                    mean_series,
                    x="Quantization Level",
                    y="Mean UR Pass Rate",
                    color="Quantization Level",
                    text_auto=".3f",
                    title="Average Pass Rate Comparison (Higher is Better)",
                    range_y=[0, 1.0]
                )
                fig_bar.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                st.markdown("#### Prompt-by-Prompt Pass Rate Trajectory")
                df_matrix["Prompt_Index"] = [f"P_{i+1:02d}" for i in range(len(df_matrix))]
                melted = df_matrix.melt(id_vars=["Prompt_Index"], var_name="Quantization", value_name="Pass Rate")
                
                fig_line = px.line(
                    melted,
                    x="Prompt_Index",
                    y="Pass Rate",
                    color="Quantization",
                    markers=True,
                    title="Per-Prompt Pass Rate (HumanEvalPlus Benchmark)"
                )
                fig_line.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_line, use_container_width=True)

            with st.expander("🔍 View Raw Evaluation Matrix JSON"):
                st.json(data)

        except Exception as e:
            st.error(f"Error loading evaluation matrix: {e}")
    else:
        st.info("Evaluation matrix (`./results/real_eval_matrix.json`) not generated yet. Run evaluation to produce results.")

    # Auto rerun loop if active process is running
    if is_running:
        time.sleep(1)
        st.rerun()


def view_stat_summary():
    st.markdown('<div class="main-header">📊 Statistical Summary & Hardware Recommender (run_real_stat)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inferential non-parametric statistical testing (Friedman $\chi^2$, Dunn Post-Hoc) and hardware matching.</div>', unsafe_allow_html=True)

    state = stats_runner.get_state()
    is_running = state["is_running"]

    col_btn1, col_status = st.columns([2, 5])
    with col_btn1:
        if st.button("⚡ Run Statistical Pipeline", type="primary", use_container_width=True):
            cmd = [sys.executable, "run_real_stats.py"]
            stats_runner.start_task(cmd, cwd=PROJECT_ROOT)
            st.rerun()
    with col_status:
        if is_running:
            st.markdown('<span class="status-badge-running">● EXECUTING STATS</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-idle">● READY</span>', unsafe_allow_html=True)

    st.markdown("---")

    matrix_file = "./results/real_eval_matrix.json"
    if not os.path.exists(matrix_file):
        st.warning("⚠️ Matrix file `./results/real_eval_matrix.json` not found. Please run the LLM Evaluation step first.")
        return

    with open(matrix_file, "r") as f:
        matrix_data = json.load(f)
    
    df_eval = pd.DataFrame(matrix_data)
    
    # Run statistical analysis directly in Streamlit for dynamic interactivity
    stat_results = analyze_quantization_impact(df_eval)

    # 1. Metric Cards Dashboard
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Friedman Chi-Squared (χ²)", f"{stat_results['friedman_stat']:.4f}")
    with col_m2:
        p_val = stat_results['p_value']
        sig_text = "p < 0.05 (Significant)" if p_val < 0.05 else "p ≥ 0.05 (Not Sig)"
        st.metric("Friedman p-value", f"{p_val:.4e}", delta=sig_text)
    with col_m3:
        st.metric("Kendall's W (Effect Size)", f"{stat_results['kendalls_w']:.4f}")
    with col_m4:
        st.metric("Degradation Elbow Point", stat_results['elbow_point'], delta="Optimal Tradeoff")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("🔥 Dunn's Post-Hoc Pairwise p-Adjusted Matrix")
        st.write("Bonferroni-adjusted p-values testing statistically significant performance degradation vs baseline (FP16):")
        
        posthoc_df = pd.DataFrame(stat_results["dunn_posthoc"])
        
        fig_heatmap = px.imshow(
            posthoc_df,
            text_auto=".4f",
            color_continuous_scale="Viridis",
            title="Post-Hoc Dunn Pairwise p-value Heatmap",
            labels=dict(x="Quantization Level", y="Quantization Level", color="p-adj Value")
        )
        fig_heatmap.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_right:
        st.subheader("💻 Hardware-Aware Recommendation Matrix")
        st.write("Select target GPU VRAM to evaluate hardware compatibility and recommended quantization:")

        user_vram = st.slider("Target GPU VRAM (GB):", min_value=1.0, max_value=32.0, value=8.0, step=0.5)

        rec = recommend_model(stat_results, user_vram_gb=user_vram)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RECOMMENDED QUANTIZATION</div>
            <div class="metric-value">{rec['recommended_quantization'] or 'NONE (VRAM Too Low)'}</div>
            <p style="color:#9CA3AF; margin-top:8px;"><b>Target VRAM:</b> {user_vram:.1f} GB</p>
            <p style="color:#9CA3AF;"><b>Required VRAM:</b> {rec['required_vram_gb'] if rec['required_vram_gb'] else 'N/A'} GB</p>
            <p style="color:#9CA3AF;"><b>Elbow Point:</b> {rec['elbow_point']}</p>
            <p style="color:#10B981;"><b>Status:</b> {rec['status']}</p>
            <p style="color:#E5E7EB;"><b>Rationale:</b> {rec['reason']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Hardware Matrix Lookup Table")
        lookup_rows = []
        for v in [16.0, 12.0, 8.0, 6.0, 5.0, 4.0]:
            r = recommend_model(stat_results, user_vram_gb=v)
            lookup_rows.append({
                "VRAM Target (GB)": f"{v:.1f} GB",
                "Recommendation": r["recommended_quantization"] or "None",
                "Required VRAM": f"{r['required_vram_gb']:.1f} GB" if r["required_vram_gb"] else "N/A",
                "Status": r["status"]
            })
        st.dataframe(pd.DataFrame(lookup_rows), use_container_width=True, hide_index=True)


def view_hardware_config():
    st.markdown('<div class="main-header">🖥️ GPU & Hardware Driver Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect system GPU hardware, configure AMD ROCm/HIP driver parameters, and manage hardware acceleration backends.</div>', unsafe_allow_html=True)

    from utils.gpu_helper import detect_gpu_hardware, configure_amd_gpu_env, AMD_GFX_TARGETS
    hw = detect_gpu_hardware()

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric("Detected Vendor", hw["vendor"], delta="Active")
    with col_h2:
        st.metric("GPU Model", hw["gpu_name"][:20])
    with col_h3:
        st.metric("VRAM Total", f"{hw['vram_total_gb']} GB")
    with col_h4:
        st.metric("Recommended Backend", hw["recommended_backend"])

    st.markdown("---")
    st.markdown("### 🛠️ AMD ROCm / HIP Driver Settings")
    st.write("Configure environment variables for AMD Radeon GPUs (RDNA2, RDNA3, Vega) running `llama-cpp-python`.")

    # Read current config if saved
    current_gfx = ""
    current_hip = "0"
    if os.path.exists("./results/eval_config.json"):
        try:
            with open("./results/eval_config.json", "r") as f:
                cfg_saved = json.load(f)
                current_gfx = cfg_saved.get("hsa_override_gfx_version", "")
                current_hip = cfg_saved.get("hip_visible_devices", "0")
        except Exception:
            pass

    col_amd1, col_amd2 = st.columns(2)
    with col_amd1:
        gfx_labels = list(AMD_GFX_TARGETS.keys())
        default_idx = 0
        for i, (k, v) in enumerate(AMD_GFX_TARGETS.items()):
            if v == current_gfx:
                default_idx = i
                break
        selected_gfx_label = st.selectbox(
            "HSA_OVERRIDE_GFX_VERSION Target:",
            options=gfx_labels,
            index=default_idx,
            help="Override GFX architecture version required for consumer AMD Radeon RX GPUs on ROCm."
        )
        selected_gfx_val = AMD_GFX_TARGETS[selected_gfx_label]
        
    with col_amd2:
        hip_device_id = st.text_input("HIP_VISIBLE_DEVICES:", value=current_hip, help="GPU index to expose to HIP/ROCm runtime.")

    if st.button("💾 Apply & Save AMD GPU Driver Configuration", type="primary"):
        configure_amd_gpu_env(gfx_version=selected_gfx_val, hip_device=hip_device_id)
        cfg = {}
        if os.path.exists("./results/eval_config.json"):
            try:
                with open("./results/eval_config.json", "r") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["hsa_override_gfx_version"] = selected_gfx_val
        cfg["hip_visible_devices"] = hip_device_id
        os.makedirs("./results", exist_ok=True)
        with open("./results/eval_config.json", "w") as f:
            json.dump(cfg, f, indent=4)
        st.success(f"✓ Applied AMD Configuration: HSA_OVERRIDE_GFX_VERSION='{selected_gfx_val}', HIP_VISIBLE_DEVICES='{hip_device_id}'")

    st.markdown("---")
    st.markdown("### 📚 Build & Installation Guide for AMD Acceleration")
    with st.expander("📖 View Installation Commands for AMD GPUs"):
        st.code("""
# 1. Install llama-cpp-python with AMD ROCm (HIP) Acceleration:
CMAKE_ARGS="-DGGML_HIPBLAS=on" HSA_OVERRIDE_GFX_VERSION=10.3.0 pip install llama-cpp-python --force-reinstall --no-cache-dir

# 2. Alternative: Install with Universal AMD Vulkan Acceleration:
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 3. Environment Variables for Terminal Sessions:
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6000 series (RDNA2)
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7000 series (RDNA3)
export HIP_VISIBLE_DEVICES=0
""", language="bash")


def main():
    selected_view = render_sidebar()

    if selected_view == "🔍 Project Overview & Scanner":
        view_project_scanner()
    elif selected_view == "📥 Model Downloader (Hugging Face)":
        view_model_downloader()
    elif selected_view == "🖥️ GPU & Hardware Config (AMD / NVIDIA)":
        view_hardware_config()
    elif selected_view == "⚡ LLM Evaluation (eval_real_llm)":
        view_eval_runner()
    elif selected_view == "📊 Statistical Summary (run_real_stat)":
        view_stat_summary()


if __name__ == "__main__":
    main()
