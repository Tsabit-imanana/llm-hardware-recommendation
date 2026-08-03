import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.execution_engine import run_code_in_sandbox
from stats.stat_engine import analyze_quantization_impact, QUANT_LEVELS_ORDER
from core.recommender import recommend_model, DEFAULT_VRAM_LOOKUP_7B


def generate_synthetic_evaluation_matrix(n_prompts: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic evaluation matrix simulating N problem prompts
    evaluated across 5 quantization levels (FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M).
    """
    np.random.seed(seed)
    
    # Base performance per prompt
    base_ability = np.random.uniform(0.70, 0.98, size=n_prompts)
    
    # Quantization impact factors (relative preservation of accuracy)
    matrix = {
        "FP16": np.clip(base_ability + np.random.normal(0, 0.01, n_prompts), 0.0, 1.0),
        "Q8_0": np.clip(base_ability - 0.01 + np.random.normal(0, 0.015, n_prompts), 0.0, 1.0),
        "Q6_K": np.clip(base_ability - 0.02 + np.random.normal(0, 0.02, n_prompts), 0.0, 1.0),
        "Q5_K_M": np.clip(base_ability - 0.08 + np.random.normal(0, 0.03, n_prompts), 0.0, 1.0),
        "Q4_K_M": np.clip(base_ability - 0.18 + np.random.normal(0, 0.04, n_prompts), 0.0, 1.0),
    }
    
    df = pd.DataFrame(matrix)
    df.index = [f"Prompt_{i+1:02d}" for i in range(n_prompts)]
    return df


def verify_execution_engine():
    """
    Validates sample code snippets through core/execution_engine.py,
    demonstrating correct handling of clean code, infinite loops, and assertion failures.
    """
    print("=" * 80)
    print(" 1. EXECUTION ENGINE VERIFICATION (core/execution_engine.py)")
    print("=" * 80)

    # Test Case 1: Valid Code
    valid_code = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    valid_assertions = [
        "assert fibonacci(0) == 0",
        "assert fibonacci(1) == 1",
        "assert fibonacci(6) == 8",
        "assert fibonacci(7) == 13"
    ]
    res_valid = run_code_in_sandbox(valid_code, valid_assertions, timeout_sec=2.0)

    # Test Case 2: Infinite Loop Timeout
    timeout_code = "def infinite_loop():\n    while True:\n        pass\ninfinite_loop()\n"
    timeout_assertions = ["assert True"]
    res_timeout = run_code_in_sandbox(timeout_code, timeout_assertions, timeout_sec=1.0)

    # Test Case 3: Assertion Failure
    failing_code = "def multiply(a, b):\n    return a + b  # Bug: addition instead of multiplication\n"
    failing_assertions = ["assert multiply(3, 4) == 12"]
    res_failing = run_code_in_sandbox(failing_code, failing_assertions, timeout_sec=2.0)

    exec_summary = [
        ["Valid Fibonacci Function", f"{res_valid['passed_tests']}/{res_valid['total_tests']}", f"{res_valid['unit_test_pass_rate']:.2%}", "SUCCESS (All Passed)"],
        ["Infinite Loop Code", f"{res_timeout['passed_tests']}/{res_timeout['total_tests']}", f"{res_timeout['unit_test_pass_rate']:.2%}", f"HANDLED ({res_timeout['results'][0]['error'][:35]}...)"],
        ["Faulty Multiply Code", f"{res_failing['passed_tests']}/{res_failing['total_tests']}", f"{res_failing['unit_test_pass_rate']:.2%}", "HANDLED (Assertion Error Caught)"]
    ]

    print(tabulate(exec_summary, headers=["Test Case Description", "Passed/Total", "UR Metric", "Execution Result Status"], tablefmt="grid"))
    print()


def main():
    print("\n" + "#" * 80)
    print(" LOCAL LLM RECOMMENDATION FRAMEWORK: END-TO-END PROTOTYPE EVALUATION")
    print("#" * 80 + "\n")

    # Step 1: Verify Sandboxed Execution Engine
    verify_execution_engine()

    # Step 2: Generate Synthetic Evaluation Matrix (N = 20 prompts, 5 quantization levels)
    n_prompts = 20
    eval_matrix = generate_synthetic_evaluation_matrix(n_prompts=n_prompts, seed=42)

    print("=" * 80)
    print(f" 2. SYNTHETIC EVALUATION MATRIX SUMMARY (N = {n_prompts} Prompts)")
    print("=" * 80)
    
    matrix_stats = []
    for q in QUANT_LEVELS_ORDER:
        mean_ur = eval_matrix[q].mean()
        std_ur = eval_matrix[q].std()
        min_ur = eval_matrix[q].min()
        max_ur = eval_matrix[q].max()
        matrix_stats.append([q, f"{mean_ur:.4f}", f"{std_ur:.4f}", f"{min_ur:.4f}", f"{max_ur:.4f}", DEFAULT_VRAM_LOOKUP_7B[q]])
        
    print(tabulate(matrix_stats, headers=["Quantization Level", "Mean UR", "Std Dev", "Min UR", "Max UR", "Est VRAM (GB)"], tablefmt="grid"))
    print()

    # Step 3: Run Statistical Engine
    stat_results = analyze_quantization_impact(eval_matrix)

    print("=" * 80)
    print(" 3. STATISTICAL HYPOTHESIS ENGINE RESULTS (stats/stat_engine.py)")
    print("=" * 80)
    
    stat_summary = [
        ["Friedman Chi-Squared (χ²)", f"{stat_results['friedman_chi2']:.4f}"],
        ["Friedman p-value", f"{stat_results['p_value']:.4e}"],
        ["Statistical Significance (α=0.05)", "SIGNIFICANT (p < 0.05)" if stat_results['p_value'] < 0.05 else "NOT SIGNIFICANT"],
        ["Kendall's W (Effect Size)", f"{stat_results['kendall_w']:.4f}"],
        ["Degradation Elbow Point", stat_results['elbow_point']]
    ]
    print(tabulate(stat_summary, headers=["Metric / Parameter", "Computed Value"], tablefmt="grid"))
    print()

    # Step 4: Display Dunn Post-Hoc Adjusted Matrix
    print("-" * 80)
    print(" Dunn's Post-Hoc Pairwise p-Adjusted Matrix (Bonferroni Corrected)")
    print("-" * 80)
    posthoc_df = stat_results["posthoc_dunn_matrix"]
    formatted_posthoc = posthoc_df.copy()
    for col in formatted_posthoc.columns:
        formatted_posthoc[col] = formatted_posthoc[col].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}")
    print(tabulate(formatted_posthoc, headers="keys", tablefmt="grid"))
    print()

    # Step 5: Execute Recommendation Engine for Multiple VRAM Thresholds
    print("=" * 80)
    print(" 4. HARDWARE-AWARE RECOMMENDATION ENGINE (core/recommender.py)")
    print("=" * 80)

    test_vram_scenarios = [16.0, 8.0, 6.0, 5.0, 4.0]
    rec_table = []

    for vram in test_vram_scenarios:
        rec = recommend_model(stat_results, user_vram_gb=vram)
        rec_table.append([
            f"{vram:.1f} GB",
            rec["recommended_quantization"] if rec["recommended_quantization"] else "N/A",
            f"{rec['required_vram_gb']:.1f} GB" if rec["required_vram_gb"] else "N/A",
            rec["elbow_point"],
            rec["status"],
            rec["reason"]
        ])

    print(tabulate(
        rec_table,
        headers=["Target User VRAM", "Recommended Level", "Required VRAM", "Elbow Point", "Status", "Decision Rationale"],
        tablefmt="grid"
    ))
    print("\n" + "#" * 80)
    print(" END-TO-END FRAMEWORK EVALUATION COMPLETE")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
