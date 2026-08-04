import os
import sys
import json
import pandas as pd
from stats.stat_engine import analyze_quantization_impact
from core.recommender import recommend_model

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

sys.stdout = TeeLogger("./logs/real_stats_summary.log")


def main():
    matrix_file = "./results/real_eval_matrix.json"
    
    try:
        with open(matrix_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {matrix_file}. Please run eval_real_llm.py first.")
        return

    df = pd.DataFrame(data)
    print("\n==========================================")
    print(" 📊 REAL EVALUATION MATRIX SUMMARY")
    print("==========================================")
    print(df.describe())
    
    # Run Inferential Statistical Pipeline
    stat_results = analyze_quantization_impact(df)
    
    print("\n==========================================")
    print(" REAL STATISTICAL RESULTS FOR PAPER (Chapter 4)")
    print("==========================================")
    print(f"Friedman Chi-Squared (χ²): {stat_results['friedman_stat']:.4f}")
    print(f"Friedman p-value:          {stat_results['p_value']:.4e}")
    print(f"Kendall's W (Effect Size): {stat_results['kendalls_w']:.4f}")
    print(f"Degradation Elbow Point:   {stat_results['elbow_point']}")
    
    print("\nDunn's Post-Hoc Pairwise p-Adjusted Matrix:")
    print(pd.DataFrame(stat_results["dunn_posthoc"]))
    
    # Run Recommendation Logic across VRAM thresholds
    print("\n==========================================")
    print(" HARDWARE-AWARE RECOMMENDATION MATRIX")
    print("==========================================")
    for vram in [16.0, 8.0, 6.0, 5.0, 4.0]:
        rec = recommend_model(stat_results, user_vram_gb=vram)
        print(f"VRAM Target: {vram:4.1f} GB -> Recommendation: {rec}")

if __name__ == "__main__":
    main()
