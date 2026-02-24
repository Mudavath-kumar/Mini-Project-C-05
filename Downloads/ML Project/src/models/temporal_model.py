"""
src/models/temporal_model.py  —  Step 4: Temporal Anomaly Detector
--------------------------------------------------------------------
Vectorised feature extraction + Isolation Forest per product.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SAVE_PATH = "models/saved/temporal_model.pkl"
SEED = 42

FEATURE_COLS = [
    "daily_mean", "daily_std", "daily_max", "daily_cv",
    "burst_days", "burst_ratio", "inter_gap_mean", "inter_gap_std",
    "days_with_reviews", "review_velocity", "rating_std", "five_star_pct",
]


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorised — no Python loop over products."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["day"] = df["date"].dt.normalize()

    # Daily counts per (product, day)
    daily = (
        df.groupby(["product_id", "day"])
        .agg(cnt=("rating", "count"), rating_mean=("rating", "mean"))
        .reset_index()
    )

    grp_daily = daily.groupby("product_id")["cnt"]
    feat = pd.DataFrame({
        "total_reviews":     df.groupby("product_id").size(),
        "daily_mean":        grp_daily.mean().round(4),
        "daily_std":         grp_daily.std(ddof=0).fillna(0).round(4),
        "daily_max":         grp_daily.max(),
        "days_with_reviews": grp_daily.count(),
    })
    feat["daily_cv"] = (feat["daily_std"] / feat["daily_mean"].clip(lower=1e-6)).round(4)

    span = daily.groupby("product_id")["day"].agg(["min", "max"])
    feat["span_days"]       = ((span["max"] - span["min"]).dt.days + 1).clip(lower=1)
    feat["review_velocity"] = (feat["total_reviews"] / feat["span_days"]).round(4)

    # Burst days
    q75   = grp_daily.quantile(0.75).rename("q75")
    daily = daily.merge(q75.reset_index(), on="product_id", how="left")
    daily["burst_threshold"] = (daily["q75"].clip(lower=1.5) * 2).clip(lower=3)
    daily["is_burst"]        = daily["cnt"] > daily["burst_threshold"]
    burst = daily.groupby("product_id")["is_burst"].sum().rename("burst_days")
    feat  = feat.join(burst)
    feat["burst_days"]  = feat["burst_days"].fillna(0).astype(int)
    feat["burst_ratio"] = (feat["burst_days"] / feat["days_with_reviews"].clip(lower=1)).round(4)

    # Inter-day gaps
    ds = daily.sort_values(["product_id", "day"])
    ds["prev_day"] = ds.groupby("product_id")["day"].shift(1)
    ds["gap"]      = (ds["day"] - ds["prev_day"]).dt.days
    gap_stats = ds.groupby("product_id")["gap"].agg(
        inter_gap_mean="mean", inter_gap_std="std"
    ).fillna(0).round(4)
    feat = feat.join(gap_stats)

    # Rating
    rg = df.groupby("product_id")["rating"]
    feat["rating_std"]    = rg.std(ddof=0).fillna(0).round(4)
    feat["five_star_pct"] = rg.apply(lambda s: (s == 5).mean()).round(4)

    feat = feat.drop(columns=["span_days"], errors="ignore")
    feat.index.name = "product_id"
    return feat


class TemporalModel:
    def __init__(self, save_path=SAVE_PATH):
        self.save_path  = save_path
        self.model      = None
        self.scaler     = None
        self.is_trained = False

    def train(self, csv_path="data/processed/clean_reviews.csv",
              contamination=0.15, verbose=True) -> dict:
        df   = pd.read_csv(csv_path)
        feat = build_temporal_features(df)
        n    = len(feat)

        if n > 10_000:
            n_est, max_samp = 50, 256
        elif n > 2_000:
            n_est, max_samp = 100, 512
        else:
            n_est, max_samp = 200, "auto"

        X = feat[FEATURE_COLS].fillna(0).values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=n_est, max_samples=max_samp,
            contamination=contamination, random_state=SEED, n_jobs=-1,
        )
        self.model.fit(Xs)
        self.is_trained = True

        preds      = self.model.predict(Xs)
        raw        = self.model.score_samples(Xs)
        n_anomaly  = (preds == -1).sum()
        anom_mask  = preds == -1

        feat_df = feat[FEATURE_COLS].fillna(0)
        metrics = {
            "n_products":  n,
            "n_anomalies": int(n_anomaly),
            "anomaly_pct": round(n_anomaly / n * 100, 1),
            "score_range": (round(float(raw.min()), 4), round(float(raw.max()), 4)),
            "n_estimators": n_est,
            "max_samples":  max_samp,
            "feat_means_anomaly": feat_df[anom_mask].mean().round(4).to_dict(),
            "feat_means_normal":  feat_df[~anom_mask].mean().round(4).to_dict(),
            "feature_cols":       FEATURE_COLS,
        }

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, self.save_path)
        if verbose:
            print(f"[TemporalModel] Products: {n} | Anomalies: {n_anomaly} ({metrics['anomaly_pct']}%)")
        return metrics

    def predict(self, csv_path="data/processed/clean_reviews.csv",
                save_path="data/features/temporal_scores.csv",
                verbose=True) -> pd.DataFrame:
        if not self.is_trained:
            self._load()
        df   = pd.read_csv(csv_path)
        feat = build_temporal_features(df)
        X    = feat[FEATURE_COLS].fillna(0).values
        Xs   = self.scaler.transform(X)
        raw  = self.model.score_samples(Xs)
        norm = (raw - raw.max()) / (raw.min() - raw.max() + 1e-9)
        norm = np.clip(norm, 0, 1)
        is_anomaly = (self.model.predict(Xs) == -1).astype(int)

        out = pd.DataFrame({
            "product_id":     feat.index,
            "temporal_score": np.round(norm, 4),
            "is_anomaly":     is_anomaly,
        })
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
        if verbose:
            print(f"[TemporalModel] Scored {len(out)} products | {is_anomaly.sum()} anomalies")
        return out

    def _load(self):
        if not Path(self.save_path).exists():
            raise FileNotFoundError(f"No model at {self.save_path}. Train first.")
        data = joblib.load(self.save_path)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = True

    @classmethod
    def load(cls, save_path=SAVE_PATH):
        obj = cls(save_path=save_path)
        obj._load()
        return obj
