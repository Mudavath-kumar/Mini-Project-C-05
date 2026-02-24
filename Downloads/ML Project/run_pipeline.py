"""
run_pipeline.py — End-to-end pipeline runner
Usage:
    python run_pipeline.py
    python run_pipeline.py --data data/raw/reviews.csv
    python run_pipeline.py --data data/raw/reviews.csv --skip-train
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fake Hype Detection Pipeline")
    parser.add_argument("--data", default=None, help="Path to input CSV")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use saved models")
    args = parser.parse_args()

    t0 = time.time()

    # Resolve data path
    data_path = args.data
    if data_path is None:
        candidates = [
            "data/raw/reviews.csv",
            "data/raw/amazon_reviews.csv",
            "data/raw/fake_reviews.csv",
        ]
        for c in candidates:
            if Path(c).exists():
                data_path = c
                break

    if data_path is None or not Path(data_path).exists():
        print("[Pipeline] No dataset found. Generating sample data ...")
        import scripts.generate_sample_data  # noqa
        data_path = "data/raw/amazon_reviews.csv"

    print(f"[Pipeline] Data: {data_path}")

    # Step 2 — Load & clean
    print("[Pipeline] Step 2: Loading & cleaning ...")
    from src.data.loader import load_and_clean
    clean_df = load_and_clean(data_path, save_path="data/processed/clean_reviews.csv")

    if not args.skip_train:
        # Step 3 — Text model
        print("[Pipeline] Step 3: Training text model ...")
        from src.models.text_model import TextModel
        tm = TextModel()
        tm.train("data/processed/clean_reviews.csv")
        tm.predict("data/processed/clean_reviews.csv",
                   save_path="data/features/text_probs.csv")

        # Step 4 — Temporal model
        print("[Pipeline] Step 4: Training temporal model ...")
        from src.models.temporal_model import TemporalModel
        tp = TemporalModel()
        tp.train("data/processed/clean_reviews.csv")
        tp.predict("data/processed/clean_reviews.csv",
                   save_path="data/features/temporal_scores.csv")
    else:
        print("[Pipeline] Skipping training (--skip-train)")

    # Step 5 — Fusion
    print("[Pipeline] Step 5: Computing hype scores ...")
    from src.models.fusion import compute_hype_scores
    compute_hype_scores()

    print(f"[Pipeline] Done in {time.time()-t0:.1f}s")
    print("[Pipeline] Launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
