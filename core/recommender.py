from stats.stat_engine import STANDARD_QUANT_PRECISION_ORDER, get_quant_order

DEFAULT_VRAM_LOOKUP_7B = {
    "F32": 28.0,
    "FP16": 14.0,
    "BF16": 14.0,
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
    elbow_point = stat_results.get("elbow_point", "FP16")
    
    # Determine quantization order dynamically from stat_results or fallback
    quant_order = stat_results.get("quant_order")
    if not quant_order:
        mean_rates = stat_results.get("mean_pass_rates", {})
        if mean_rates:
            quant_order = get_quant_order(list(mean_rates.keys()))
        else:
            quant_order = list(STANDARD_QUANT_PRECISION_ORDER)
    else:
        quant_order = list(quant_order)

    if elbow_point not in quant_order:
        quant_order.append(elbow_point)
        quant_order = get_quant_order(quant_order)

    elbow_idx = quant_order.index(elbow_point)

    # Build dynamic vram_lookup if not provided or missing keys
    lookup_map = dict(vram_lookup) if vram_lookup else {}

    # Gather local model file sizes for dynamic VRAM estimation
    local_models_vram = {}
    try:
        import os
        from utils.hf_helper import get_local_models
        for m in get_local_models():
            vram_est = round(m["size_gb"] + 1.0, 1)
            stem = os.path.splitext(m["filename"])[0]
            local_models_vram[stem] = vram_est
    except Exception:
        pass

    for q in quant_order:
        if q not in lookup_map:
            if q in DEFAULT_VRAM_LOOKUP_7B:
                lookup_map[q] = DEFAULT_VRAM_LOOKUP_7B[q]
            elif q in local_models_vram:
                lookup_map[q] = local_models_vram[q]
            else:
                lookup_map[q] = 5.0

    # Filter candidate quantization levels satisfying Condition 2: Precision level >= Elbow Point
    valid_precision_candidates = [
        q for q in quant_order
        if q in lookup_map and quant_order.index(q) <= elbow_idx
    ]

    if not valid_precision_candidates:
        valid_precision_candidates = [elbow_point]

    # Filter candidate quantization levels satisfying Condition 1: V_quant <= V_user
    feasible_candidates = [
        q for q in valid_precision_candidates
        if lookup_map.get(q, 999.0) <= user_vram_gb
    ]

    if feasible_candidates:
        # Objective: minimize VRAM usage among feasible candidates
        best_candidate = min(feasible_candidates, key=lambda q: lookup_map[q])
        status = "OPTIMAL"
        recommendation_reason = (
            f"Quantization '{best_candidate}' minimizes VRAM footprint ({lookup_map[best_candidate]} GB) "
            f"while satisfying available VRAM ({user_vram_gb} GB) and maintaining statistical non-degradation "
            f"(Precision >= Elbow Point '{elbow_point}')."
        )
    else:
        # Fallback handling when hardware constraint cannot be met with non-degraded models
        status = "INSUFFICIENT_VRAM"
        min_vram_required = min(lookup_map.get(q, 999.0) for q in valid_precision_candidates)
        best_candidate = None
        recommendation_reason = (
            f"Insufficient VRAM: User has {user_vram_gb} GB VRAM, but minimum VRAM required for non-degraded "
            f"accuracy (Precision >= Elbow Point '{elbow_point}') is {min_vram_required} GB."
        )

    return {
        "status": status,
        "recommended_quantization": best_candidate,
        "required_vram_gb": lookup_map.get(best_candidate) if best_candidate else None,
        "user_vram_gb": user_vram_gb,
        "elbow_point": elbow_point,
        "feasible_candidates": feasible_candidates,
        "all_valid_precision_levels": valid_precision_candidates,
        "vram_lookup": lookup_map,
        "reason": recommendation_reason
    }

