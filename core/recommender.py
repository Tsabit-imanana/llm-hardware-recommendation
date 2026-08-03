from stats.stat_engine import QUANT_LEVELS_ORDER

DEFAULT_VRAM_LOOKUP_7B = {
    "FP16": 14.0,
    "Q8_0": 7.5,
    "Q6_K": 6.0,
    "Q5_K_M": 5.0,
    "Q4_K_M": 4.2
}

def recommend_model(
    stat_results: dict,
    user_vram_gb: float,
    vram_lookup: dict = None
) -> dict:
    """
    Hardware-aware recommendation algorithm corresponding to Section 3.4 of the manuscript.

    Recommends the quantization level that minimizes VRAM usage subject to:
      1. V_quant <= V_user (Hardware VRAM constraint)
      2. Quantization precision level >= Elbow Point (Statistical non-degradation threshold)

    Args:
        stat_results (dict): Output from stats.stat_engine.analyze_quantization_impact.
        user_vram_gb (float): User's available GPU VRAM in GB.
        vram_lookup (dict, optional): Map of quantization levels to VRAM requirements in GB.

    Returns:
        dict: Recommendation decision including optimal level, required VRAM, status, and eligible options.
    """
    if vram_lookup is None:
        vram_lookup = DEFAULT_VRAM_LOOKUP_7B

    elbow_point = stat_results.get("elbow_point", "FP16")
    
    if elbow_point not in QUANT_LEVELS_ORDER:
        raise ValueError(f"Elbow point '{elbow_point}' not recognized in standard quantization order.")

    elbow_idx = QUANT_LEVELS_ORDER.index(elbow_point)

    # Filter candidate quantization levels satisfying Condition 2: Precision level >= Elbow Point
    # Index <= elbow_idx means precision is higher or equal to elbow point
    valid_precision_candidates = [
        q for q in QUANT_LEVELS_ORDER
        if q in vram_lookup and QUANT_LEVELS_ORDER.index(q) <= elbow_idx
    ]

    # Filter candidate quantization levels satisfying Condition 1: V_quant <= V_user
    feasible_candidates = [
        q for q in valid_precision_candidates
        if vram_lookup[q] <= user_vram_gb
    ]

    if feasible_candidates:
        # Objective: minimize VRAM usage among feasible candidates
        # Sort by VRAM requirement ascending (or precision descending)
        best_candidate = min(feasible_candidates, key=lambda q: vram_lookup[q])
        status = "OPTIMAL"
        recommendation_reason = (
            f"Quantization '{best_candidate}' minimizes VRAM footprint ({vram_lookup[best_candidate]} GB) "
            f"while satisfying available VRAM ({user_vram_gb} GB) and maintaining statistical non-degradation "
            f"(Precision >= Elbow Point '{elbow_point}')."
        )
    else:
        # Fallback handling when hardware constraint cannot be met with non-degraded models
        status = "INSUFFICIENT_VRAM"
        # Find minimum required VRAM for non-degraded performance
        min_vram_required = min(vram_lookup[q] for q in valid_precision_candidates)
        best_candidate = None
        recommendation_reason = (
            f"Insufficient VRAM: User has {user_vram_gb} GB VRAM, but minimum VRAM required for non-degraded "
            f"accuracy (Precision >= Elbow Point '{elbow_point}') is {min_vram_required} GB."
        )

    return {
        "status": status,
        "recommended_quantization": best_candidate,
        "required_vram_gb": vram_lookup.get(best_candidate) if best_candidate else None,
        "user_vram_gb": user_vram_gb,
        "elbow_point": elbow_point,
        "feasible_candidates": feasible_candidates,
        "all_valid_precision_levels": valid_precision_candidates,
        "vram_lookup": vram_lookup,
        "reason": recommendation_reason
    }
