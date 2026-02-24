"""
src/models/text_model.py  —  Step 3: TF-IDF + Logistic Regression
------------------------------------------------------------------
Trains a fake review classifier on cleaned review text.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve,
)

SAVE_PATH = "models/saved/text_model.pkl"
SEED = 42


class TextModel:
    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path  = save_path
        self.pipeline   = None
        self.is_trained = False
        self.has_labels = False

    def _build_pipeline(self):
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=SEED,
            )),
        ])

    def train(self, csv_path="data/processed/clean_reviews.csv",
              text_col="text_clean", label_col="is_fake",
              test_size=0.2, verbose=True) -> dict:

        df = pd.read_csv(csv_path)
        labeled = df[df[label_col] >= 0].copy() if label_col in df.columns else pd.DataFrame()

        if len(labeled) >= 20:
            self.has_labels = True
            X = labeled[text_col].fillna("").astype(str)
            y = labeled[label_col].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=SEED
            )
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X_train, y_train)

            y_pred = self.pipeline.predict(X_test)
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]
            auc    = roc_auc_score(y_test, y_prob)
            cm     = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            report = classification_report(y_test, y_pred,
                        target_names=["Genuine", "Fake"], output_dict=True)

            tfidf = self.pipeline.named_steps["tfidf"]
            clf   = self.pipeline.named_steps["clf"]
            names = tfidf.get_feature_names_out()
            coefs = clf.coef_[0]
            top_fake    = [{"word": names[i], "score": round(float(coefs[i]), 4)}
                           for i in coefs.argsort()[-25:][::-1]]
            top_genuine = [{"word": names[i], "score": round(float(coefs[i]), 4)}
                           for i in coefs.argsort()[:25]]

            metrics = {
                "mode": "supervised",
                "train_n": len(X_train), "test_n": len(X_test),
                "roc_auc": round(auc, 4),
                "report": report,
                "confusion_matrix": cm.tolist(),
                "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
                "top_fake_words": top_fake,
                "top_genuine_words": top_genuine,
            }

            if verbose:
                print(f"[TextModel] ROC-AUC: {auc:.4f}")
                print(classification_report(y_test, y_pred,
                      target_names=["Genuine", "Fake"]))
        else:
            self.has_labels = False
            X = df[text_col].fillna("").astype(str)

            def heuristic_label(t):
                score = 0
                if t.upper() == t and len(t) > 5: score += 2
                if t.count("!") >= 2: score += 1
                if len(t.split()) <= 5: score += 1
                if sum(1 for w in t.split() if w.isupper() and len(w) > 2) >= 2: score += 1
                return int(score >= 3)

            y_proxy = X.apply(heuristic_label)
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X, y_proxy)
            metrics = {"mode": "heuristic (no labels found)"}
            if verbose:
                print("[TextModel] Mode: heuristic (no is_fake labels)")

        self.is_trained = True
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "has_labels": self.has_labels},
                    self.save_path)
        if verbose:
            print(f"[TextModel] Saved: {self.save_path}")
        return metrics

    def predict(self, csv_path="data/processed/clean_reviews.csv",
                text_col="text_clean",
                save_path="data/features/text_probs.csv",
                verbose=True) -> pd.DataFrame:
        if not self.is_trained:
            self._load()
        df   = pd.read_csv(csv_path)
        prob = self.pipeline.predict_proba(df[text_col].fillna("").astype(str))[:, 1]
        out  = pd.DataFrame({
            "review_id":  df["review_id"],
            "product_id": df["product_id"],
            "fake_prob":  np.round(prob, 4),
        })
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
        if verbose:
            print(f"[TextModel] Scored {len(out):,} reviews | mean fake_prob={prob.mean():.4f}")
        return out

    def _load(self):
        if not Path(self.save_path).exists():
            raise FileNotFoundError(f"No model at {self.save_path}. Train first.")
        data = joblib.load(self.save_path)
        self.pipeline   = data["pipeline"]
        self.has_labels = data["has_labels"]
        self.is_trained = True

    @classmethod
    def load(cls, save_path=SAVE_PATH):
        obj = cls(save_path=save_path)
        obj._load()
        return obj
