import os
import re
import json
import joblib
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from typing import List, Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors

# --------------------------
# Config
# --------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DEFAULT_DATA_PATH = "dataset/ghana_jobs_detailed.csv"    # adjust if needed
ARTIFACT_DIR = "models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# --------------------------
# Utilities
# --------------------------
EN_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself
no nor not of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with you your yours yourself yourselves
""".split())

def clean_text_basic(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r'[\n\r\t]+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    tokens = [w for w in t.split() if w not in EN_STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def coalesce_text(row: pd.Series) -> str:
    """
    Prefer detailed description; fallback to details (list-ish); then title/name.
    """
    jd = row.get('job_description', None)
    if isinstance(jd, str) and jd.strip():
        return jd
    det = row.get('details', None)
    if isinstance(det, str) and det.strip():
        det_clean = re.sub(r"[\[\]']", " ", det)
        if len(det_clean.strip()) > 0:
            return det_clean
    parts = []
    for k in ('title', 'name'):
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " ".join(parts) if parts else ""

def extract_job_function(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    s = str(s)
    if ":" in s:
        return s.split(":", 1)[1].strip() or None
    return s.strip() or None

KW_MAP = {
    'sales marketing': 'Sales/Marketing',
    'marketing': 'Sales/Marketing',
    'sales': 'Sales/Marketing',
    'customer service': 'Customer Service',
    'teacher teaching education school': 'Education',
    'security safety hse': 'Health & Safety',
    'driver transport logistics dispatch': 'Driver & Transport',
    'software developer engineer programmer it data ict': 'IT/Tech',
    'account finance auditor bookkeeping': 'Finance/Accounting',
    'admin administrative receptionist office': 'Admin/Office',
    'chef cook kitchen hospitality hotel restaurant': 'Hospitality',
    'mechanic maintenance technician electrical civil': 'Engineering/Technical',
    'nurse medical clinic hospital healthcare': 'Healthcare',
}

def infer_category(row: pd.Series) -> str:
    jf = row.get('job_function_norm', None)
    ind = row.get('Industry_norm', None)
    if isinstance(jf, str) and jf.strip():
        return jf
    if isinstance(ind, str) and ind.strip():
        return ind
    text = " ".join([str(row.get('title', "")), str(row.get('name', "")), str(row.get('clean_text', ""))]).lower()
    best_cat, best_hits = None, 0
    for key_group, cat in KW_MAP.items():
        hits = sum(1 for kw in key_group.split() if kw in text)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat if best_cat else "Other/Unknown"

def safe_stratify(y: List[str], test_size: float = 0.2) -> Optional[List[str]]:
    """
    Returns y for stratify if each class has at least 2 samples for a split; else None.
    """
    cnt = Counter(y)
    min_class = min(cnt.values()) if cnt else 0
    return y if min_class >= 2 else None

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, path: str):
    plt.figure(figsize=(max(6, len(classes)*0.6), max(5, len(classes)*0.45)))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# --------------------------
# Model 1: TF-IDF + Cosine Similarity (retrieval baseline)
# --------------------------
class TfidfCosineRetriever:
    """
    Retrieval-style baseline: Fit TF-IDF on training set; for evaluation,
    predict label of nearest training doc by cosine similarity (1-NN).
    """
    def __init__(self, max_features=20000, ngram_range=(1,2), min_df=1):
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          ngram_range=ngram_range,
                                          min_df=min_df)
        self.nn = None
        self.y_train = None

    def fit(self, X_train: List[str], y_train: List[str]):
        Xv = self.vectorizer.fit_transform(X_train)
        # Cosine distance = 1 - cosine_similarity → metric='cosine'
        self.nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn.fit(Xv)
        self.y_train = np.array(y_train)

    def predict(self, X_test: List[str]) -> List[str]:
        Xv = self.vectorizer.transform(X_test)
        dist, idx = self.nn.kneighbors(Xv, n_neighbors=1)
        return self.y_train[idx[:,0]].tolist()

    def save(self, path_prefix: str):
        joblib.dump(self.vectorizer, f"{path_prefix}_vectorizer.joblib")
        joblib.dump(self.nn, f"{path_prefix}_nn.joblib")
        joblib.dump(self.y_train, f"{path_prefix}_ytrain.joblib")

# --------------------------
# Model 2: TF-IDF + Logistic Regression (supervised)
# --------------------------
def train_tfidf_logreg(X_train, y_train):
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1)
    Xv = vect.fit_transform(X_train)
    classes = np.unique(y_train)
    try:
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=classes,
                                             y=y_train)
        cw = {c:w for c,w in zip(classes, class_weights)}
    except Exception:
        cw = None
    clf = LogisticRegression(max_iter=2000, n_jobs=None, class_weight=cw)
    clf.fit(Xv, y_train)
    return vect, clf

# --------------------------
# Model 3: DistilBERT/MiniLM embeddings + Logistic Regression (supervised)
# --------------------------
def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        # MiniLM is small and strong; auto-downloads on first run
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        print(f"[Info] sentence-transformers not available ({e}). Skipping embedding model.")
        return None

def embed_texts(st_model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    # st_model.encode returns np.ndarray
    return st_model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

def train_embed_logreg(embeddings: np.ndarray, y_train: List[str]):
    classes = np.unique(y_train)
    try:
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=classes,
                                             y=y_train)
        cw = {c:w for c,w in zip(classes, class_weights)}
    except Exception:
        cw = None
    clf = LogisticRegression(max_iter=2000, class_weight=cw)
    clf.fit(embeddings, y_train)
    return clf

# --------------------------
# Main training routine
# --------------------------
def main(args):
    # Load data
    df = pd.read_csv(args.data_path)

    # Build robust text & label
    df['raw_text'] = df.apply(coalesce_text, axis=1).astype(str)
    df['clean_text'] = df['raw_text'].apply(clean_text_basic)
    df['job_function_norm'] = df['job_function'].apply(extract_job_function)
    df['Industry_norm'] = df['Industry'].apply(lambda x: x if isinstance(x, str) and x.strip() else None)
    df['category'] = df.apply(infer_category, axis=1)

    # Keep only rows with some text + label
    keep = (df['clean_text'].str.len() > 0) & (df['category'].notna())
    df = df[keep].copy().reset_index(drop=True)

    if len(df) < 10:
        print(f"[Warn] Very small dataset after cleaning: {len(df)} rows. Results may be unstable.")

    X = df['clean_text'].tolist()
    y = df['category'].tolist()

    strat = safe_stratify(y, test_size=args.test_size)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_SEED, stratify=strat
    )

    label_encoder = LabelEncoder()
    y_tr_enc = label_encoder.fit_transform(y_tr)
    y_te_enc = label_encoder.transform(y_te)
    classes = list(label_encoder.classes_)
    joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

    summary_rows = []

    # ------------------ Model 1: TF-IDF + Cosine ------------------
    print("\n[Model 1] TF-IDF + Cosine Similarity (retrieval baseline)")
    retr = TfidfCosineRetriever(max_features=20000, ngram_range=(1,2))
    retr.fit(X_tr, y_tr)
    y_pred = retr.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f} | F1-weighted: {f1_weighted:.4f}")

    # Save artifacts
    retr.save(os.path.join(ARTIFACT_DIR, "tfidf_cosine"))

    # Save report (note: no confusion matrix for retrieval baseline)
    summary_rows.append({
        "model": "TFIDF+Cosine",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    })

    # ------------------ Model 2: TF-IDF + Logistic Regression ------------------
    print("\n[Model 2] TF-IDF + Logistic Regression")
    vect2, clf2 = train_tfidf_logreg(X_tr, y_tr)
    X_te_vec = vect2.transform(X_te)
    y_pred2 = clf2.predict(X_te_vec)

    acc2 = accuracy_score(y_te, y_pred2)
    f1_macro2 = f1_score(y_te, y_pred2, average='macro', zero_division=0)
    f1_weighted2 = f1_score(y_te, y_pred2, average='weighted', zero_division=0)
    print(f"Accuracy: {acc2:.4f} | F1-macro: {f1_macro2:.4f} | F1-weighted: {f1_weighted2:.4f}")
    print("\nClassification Report (TFIDF+LR):\n", classification_report(y_te, y_pred2, zero_division=0))

    # Confusion matrix
    cm2 = confusion_matrix(y_te, y_pred2, labels=classes)
    cm2_path = os.path.join(REPORTS_DIR, "cm_tfidf_lr.png")
    plot_confusion_matrix(cm2, classes, "Confusion Matrix - TFIDF+LR", cm2_path)

    # Save artifacts
    joblib.dump(vect2, os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf2, os.path.join(ARTIFACT_DIR, "logreg_tfidf.joblib"))

    summary_rows.append({
        "model": "TFIDF+LogReg",
        "accuracy": acc2,
        "f1_macro": f1_macro2,
        "f1_weighted": f1_weighted2
    })

    # ------------------ Model 3: DistilBERT (MiniLM) embeddings + LR ------------------
    print("\n[Model 3] DistilBERT/MiniLM embeddings + Logistic Regression")
    st_model = load_sentence_transformer()
    if st_model is not None:
        X_tr_emb = embed_texts(st_model, X_tr)
        X_te_emb = embed_texts(st_model, X_te)
        clf3 = train_embed_logreg(X_tr_emb, y_tr)
        y_pred3 = clf3.predict(X_te_emb)

        acc3 = accuracy_score(y_te, y_pred3)
        f1_macro3 = f1_score(y_te, y_pred3, average='macro', zero_division=0)
        f1_weighted3 = f1_score(y_te, y_pred3, average='weighted', zero_division=0)
        print(f"Accuracy: {acc3:.4f} | F1-macro: {f1_macro3:.4f} | F1-weighted: {f1_weighted3:.4f}")
        print("\nClassification Report (MiniLM+LR):\n", classification_report(y_te, y_pred3, zero_division=0))

        # Confusion matrix
        cm3 = confusion_matrix(y_te, y_pred3, labels=classes)
        cm3_path = os.path.join(REPORTS_DIR, "cm_minilm_lr.png")
        plot_confusion_matrix(cm3, classes, "Confusion Matrix - MiniLM+LR", cm3_path)

        # Save artifacts
        joblib.dump(clf3, os.path.join(ARTIFACT_DIR, "logreg_minilm.joblib"))
        # Save a tiny wrapper config to know what ST model was used
        with open(os.path.join(ARTIFACT_DIR, "st_model.json"), "w") as f:
            json.dump({"name": "sentence-transformers/all-MiniLM-L6-v2"}, f)

        summary_rows.append({
            "model": "MiniLM+LogReg",
            "accuracy": acc3,
            "f1_macro": f1_macro3,
            "f1_weighted": f1_weighted3
        })
    else:
        print("Skipping Model 3 (MiniLM) because sentence-transformers is unavailable.")

    # ------------------ Summary table ------------------
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["f1_macro","accuracy"], ascending=False)
    sum_path = os.path.join(REPORTS_DIR, "model_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print("\n=== Summary ===")
    print(summary_df)
    print(f"\nSaved artifacts → {ARTIFACT_DIR}/")
    print(f"Saved reports   → {REPORTS_DIR}/ (confusion matrices, summary CSV)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Evaluate Job Classification Models")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to CSV (scraped dataset)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    args = parser.parse_args()
    main(args)
