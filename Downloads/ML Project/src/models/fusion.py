"""
src/models/fusion.py  —  Step 5: Hype Score Fusion
---------------------------------------------------
Combines text + temporal signals into a 0-100 Hype Score.

Weights:
    text_score     0.45
    temporal_score 0.35
    five_star_pct  0.10
    burst_ratio    0.10
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.models.temporal_model import build_temporal_features


WEIGHTS = {
    "text_score":     0.45,
    "temporal_score": 0.35,
    "five_star_pct":  0.10,
    "burst_ratio":    0.10,
}


def compute_hype_scores(
    text_probs_path="data/features/text_probs.csv",
    temporal_scores_path="data/features/temporal_scores.csv",
    clean_csv_path="data/processed/clean_reviews.csv",
    save_path="data/features/hype_scores.csv",
    verbose=True,
) -> pd.DataFrame:

    text_df = pd.read_csv(text_probs_path)
    temp_df = pd.read_csv(temporal_scores_path)
    clean_df = pd.read_csv(clean_csv_path)

    # Per-product mean fake probability
    text_prod = text_df.groupby("product_id")["fake_prob"].mean().reset_index()
    text_prod.columns = ["product_id", "text_score"]

    # Temporal features for five_star_pct and burst_ratio
    feat = build_temporal_features(clean_df).reset_index()

    merged = (
        text_prod
        .merge(temp_df[["product_id", "temporal_score", "is_anomaly"]], on="product_id", how="inner")
        .merge(feat[["product_id", "five_star_pct", "burst_ratio",
                      "total_reviews", "daily_mean", "daily_max"]], on="product_id", how="left")
    )

    merged["hype_score"] = (
        merged["text_score"]     * WEIGHTS["text_score"] +
        merged["temporal_score"] * WEIGHTS["temporal_score"] +
        merged["five_star_pct"]  * WEIGHTS["five_star_pct"] +
        merged["burst_ratio"]    * WEIGHTS["burst_ratio"]
    ) * 100

    merged["hype_score"] = merged["hype_score"].clip(0, 100).round(2)

    merged["risk_level"] = pd.cut(
        merged["hype_score"],
        bins=[-1, 32, 65, 100],
        labels=["Low", "Medium", "High"],
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(save_path, index=False)

    if verbose:
        print(f"[Fusion] Products scored : {len(merged)}")
        print(f"[Fusion] High Risk       : {(merged['risk_level']=='High').sum()}")
        print(f"[Fusion] Medium Risk     : {(merged['risk_level']=='Medium').sum()}")
        print(f"[Fusion] Low Risk        : {(merged['risk_level']=='Low').sum()}")
        print(f"[Fusion] Saved to        : {save_path}")

    return merged
