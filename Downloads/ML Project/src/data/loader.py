"""
src/data/loader.py  —  Step 2: Data Loader & Cleaner
------------------------------------------------------
Loads any review CSV and normalises column names so the
rest of the pipeline always sees the same schema:

    review_id, product_id, reviewer_id,
    review_text, text_clean, rating, date, is_fake
"""

import re
import string
import hashlib
import pandas as pd
from pathlib import Path


# ── Column aliases ────────────────────────────────────────────
# Maps common raw column names → standard internal names
COL_MAP = {
    # product
    "asin":           "product_id",
    "product":        "product_id",
    "prod_id":        "product_id",
    "productid":      "product_id",
    # review text
    "reviewtext":     "review_text",
    "review":         "review_text",
    "text":           "review_text",
    "text_":          "review_text",
    "body":           "review_text",
    "content":        "review_text",
    "comment":        "review_text",
    # rating
    "overall":        "rating",
    "stars":          "rating",
    "score":          "rating",
    # date
    "reviewtime":     "date",
    "unixreviewtime": "date",
    "review_date":    "date",
    # reviewer
    "reviewerid":     "reviewer_id",
    "user_id":        "reviewer_id",
    "userid":         "reviewer_id",
    # label
    "label":          "is_fake",
    "fake":           "is_fake",
    "isfake":         "is_fake",
    "is_fake":        "is_fake",
    # category
    "category":       "category",
}

STOP_WORDS = {
    "the","a","an","is","it","in","on","at","to","and","or","but",
    "of","for","with","this","that","was","are","be","as","by","from",
    "have","had","has","he","she","we","they","you","i","my","your",
    "our","their","its","not","no","so","if","do","did","does","will",
    "would","could","should","may","might","just","very","also","been",
    "than","then","when","where","who","which","what","how","all","any",
    "more","most","some","such","up","out","about","into","over","after",
    "there","here","can",
}


def clean_text(text: str) -> str:
    """Basic text normalisation: lowercase, remove punctuation, stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def _normalise_label(series: pd.Series) -> pd.Series:
    """Convert OR/CG strings or float labels to 0/1/-1."""
    s = series.copy().astype(str).str.strip().str.upper()
    mapping = {
        "CG": 1, "FAKE": 1, "1": 1, "1.0": 1, "TRUE": 1,
        "OR": 0, "GENUINE": 0, "REAL": 0, "0": 0, "0.0": 0, "FALSE": 0,
    }
    return s.map(mapping).fillna(-1).astype(int)


def load_and_clean(
    csv_path: str,
    save_path: str = "data/processed/clean_reviews.csv",
    min_text_length: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a raw review CSV, normalise columns, clean text.

    Returns
    -------
    DataFrame with standardised schema, saved to save_path.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ── Read ──────────────────────────────────────────────────
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # ── Normalise column names ─────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    # ── Ensure required columns exist ────────────────────────
    if "review_text" not in df.columns:
        text_candidates = [c for c in df.columns if "text" in c or "review" in c or "comment" in c]
        if text_candidates:
            df = df.rename(columns={text_candidates[0]: "review_text"})
        else:
            raise ValueError(
                f"Cannot find a review text column. Columns found: {list(df.columns)}"
            )

    if "product_id" not in df.columns:
        id_candidates = [c for c in df.columns if "product" in c or "asin" in c or "id" in c]
        if id_candidates:
            df = df.rename(columns={id_candidates[0]: "product_id"})
        else:
            df["product_id"] = "product_unknown"

    if "rating" not in df.columns:
        df["rating"] = 3.0

    if "date" not in df.columns:
        df["date"] = pd.Timestamp("2020-01-01")

    if "reviewer_id" not in df.columns:
        df["reviewer_id"] = "reviewer_unknown"

    # ── Clean text ────────────────────────────────────────────
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["text_clean"]  = df["review_text"].apply(clean_text)

    # ── Drop short / empty reviews ────────────────────────────
    df = df[df["text_clean"].str.len() >= min_text_length].copy()

    # ── Normalise rating ──────────────────────────────────────
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)
    df["rating"] = df["rating"].clip(1, 5)

    # ── Normalise date ────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].fillna(pd.Timestamp("2020-01-01"))

    # ── Label ─────────────────────────────────────────────────
    if "is_fake" in df.columns:
        df["is_fake"] = _normalise_label(df["is_fake"])
    else:
        df["is_fake"] = -1   # unknown

    # ── Unique review ID ──────────────────────────────────────
    df = df.reset_index(drop=True)
    df["review_id"] = df.index.map(lambda i: f"r{i:07d}")

    # ── Keep only needed columns (+ any extras) ───────────────
    core = ["review_id","product_id","reviewer_id",
            "review_text","text_clean","rating","date","is_fake"]
    extra = [c for c in df.columns if c not in core]
    df = df[core + extra]

    # ── Save ──────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    if verbose:
        labeled = (df["is_fake"] >= 0).sum()
        print(f"[Loader] Loaded   : {len(df):,} reviews")
        print(f"[Loader] Products : {df['product_id'].nunique():,}")
        print(f"[Loader] Labeled  : {labeled:,} reviews have is_fake")
        print(f"[Loader] Saved to : {save_path}")

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Basic validation checks — returns a dict of issues found."""
    issues = {}
    if df["review_text"].str.len().mean() < 10:
        issues["short_texts"] = "Average review text is very short"
    if df["rating"].isnull().sum() > 0:
        issues["null_ratings"] = f"{df['rating'].isnull().sum()} null ratings"
    if df["product_id"].nunique() < 2:
        issues["single_product"] = "Only one product found"
    return issues
