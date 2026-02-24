# src/utils/helpers.py
# ─────────────────────────────────────────────────────────────
# Utility functions: preprocessing, saving/loading models,
# train/test splitting, and result formatting.
# ─────────────────────────────────────────────────────────────

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import DATA_CONFIG, FEATURE_CONFIG, PLOTS_DIR, REPORTS_DIR


def split_features_target(df: pd.DataFrame):
    """Split DataFrame into X (features) and y (target)."""
    target = FEATURE_CONFIG['target_col']
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols


def train_test_split_stratified(X: pd.DataFrame, y: pd.Series):
    """
    Stratified train/test split using config settings.
    Returns X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_seed'],
        stratify=y
    )


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on train, transform both splits.
    Returns scaled arrays + fitted scaler.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def save_model(model, name: str, folder: str = "outputs"):
    """Pickle a trained model to disk."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name.replace(' ', '_').lower()}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Utils] Model saved → {path}")
    return path


def load_model(path: str):
    """Load a pickled model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[Utils] Model loaded from {path}")
    return model


def save_report(df: pd.DataFrame, filename: str):
    """Save a DataFrame as CSV to the reports directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[Utils] Report saved → {path}")
    return path


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in [PLOTS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def print_section(title: str, width: int = 60):
    """Pretty-print a section divider."""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def align_columns(df_new: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Align a new DataFrame's columns to match training feature columns.
    Adds missing columns as 0, drops extra columns.
    """
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0
    return df_new[feature_cols]
