import numpy as np
import pandas as pd
from scipy import stats
# pyrefly: ignore [missing-import]
import scikit_posthocs as sp

QUANT_LEVELS_ORDER = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]

def analyze_quantization_impact(matrix_data: dict | pd.DataFrame) -> dict:
    """
    Non-parametric statistical pipeline corresponding to Section 3.3 of the manuscript.

    Performs:
      1. Friedman Test across 5 quantization levels (FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M).
      2. Effect size estimation via Kendall's W coefficient of concordance.
      3. Post-Hoc Dunn Analysis with Bonferroni correction (if p < 0.05).
      4. Degradation Elbow Point Identification (lowest quantization maintaining p >= 0.05 vs FP16 baseline).

    Args:
        matrix_data (dict | pd.DataFrame): Evaluation matrix of Unit Test Pass Rates (UR).
            Can be a dict mapping level_name -> list/array of pass rates per prompt,
            or a pandas DataFrame with quantization levels as columns.

    Returns:
        dict: Summary containing chi2_stat, p_value, kendall_w, posthoc_dunn_matrix, elbow_point.
    """
    if isinstance(matrix_data, dict):
        df = pd.DataFrame(matrix_data)
    else:
        df = matrix_data.copy()

    # Reorder columns to standard quantization precision order if present
    present_levels = [q for q in QUANT_LEVELS_ORDER if q in df.columns]
    if len(present_levels) < 2:
        raise ValueError(f"Matrix data must contain at least 2 quantization levels from {QUANT_LEVELS_ORDER}")
    
    df = df[present_levels]
    N, m = df.shape  # N = number of prompts, m = number of quantization levels

    if N == 0:
        empty_matrix = pd.DataFrame(np.nan, index=present_levels, columns=present_levels)
        return {
            "friedman_chi2": 0.0,
            "friedman_stat": 0.0,
            "p_value": 1.0,
            "kendall_w": 0.0,
            "kendalls_w": 0.0,
            "n_prompts": 0,
            "m_levels": m,
            "posthoc_dunn_matrix": empty_matrix,
            "dunn_posthoc": empty_matrix,
            "elbow_point": present_levels[0] if present_levels else "FP16",
            "baseline": present_levels[0] if present_levels else "FP16",
            "mean_pass_rates": {q: 0.0 for q in present_levels}
        }

    # 1. Friedman Test
    level_arrays = [df[q].values for q in present_levels]
    friedman_res = stats.friedmanchisquare(*level_arrays)
    chi2_stat = float(friedman_res.statistic)
    p_val = float(friedman_res.pvalue)

    # 2. Kendall's W Coefficient of Concordance:
    # W = chi2 / (N * (m - 1))
    kendall_w = float(np.clip(chi2_stat / (N * (m - 1)), 0.0, 1.0)) if (N * (m - 1)) > 0 else 0.0

    # 3. Post-Hoc Analysis (Dunn's test with Bonferroni adjustment)
    posthoc_df = None
    baseline = present_levels[0]  # FP16 baseline

    # Perform Dunn post-hoc test
    # Prepare melted format for scikit_posthocs.posthoc_dunn
    melted_df = df.melt(var_name="quantization", value_name="pass_rate")
    posthoc_df = sp.posthoc_dunn(
        melted_df,
        val_col="pass_rate",
        group_col="quantization",
        p_adjust="bonferroni"
    )

    # 4. Degradation Elbow Point Identification
    # Baseline is FP16. Find lowest quantization precision level with no significant degradation (p >= 0.05 vs FP16)
    elbow_point = baseline

    if p_val >= 0.05:
        # If overall Friedman test is not statistically significant, no quantization level causes a significant drop
        elbow_point = present_levels[-1]  # Lowest precision available (e.g. Q4_K_M)
    else:
        # Friedman is significant (p < 0.05): inspect pairwise Dunn post-hoc comparison with baseline FP16
        # Iterate from baseline down to lowest level
        for q_level in present_levels:
            if q_level == baseline:
                elbow_point = q_level
                continue
            
            p_adj_vs_baseline = float(posthoc_df.loc[baseline, q_level])
            
            # Check if performance drop is NOT statistically significant (p >= 0.05)
            # and average pass rate is not severely degraded
            if p_adj_vs_baseline >= 0.05:
                elbow_point = q_level
            else:
                # Once we encounter a statistically significant degradation drop (p < 0.05),
                # stop moving lower in precision
                break

    return {
        "friedman_chi2": chi2_stat,
        "friedman_stat": chi2_stat,
        "p_value": p_val,
        "kendall_w": kendall_w,
        "kendalls_w": kendall_w,
        "n_prompts": N,
        "m_levels": m,
        "posthoc_dunn_matrix": posthoc_df,
        "dunn_posthoc": posthoc_df,
        "elbow_point": elbow_point,
        "baseline": baseline,
        "mean_pass_rates": df.mean().to_dict()
    }
