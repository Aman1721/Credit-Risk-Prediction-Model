# src/evaluation/evaluator.py
# ─────────────────────────────────────────────────────────────
# Model evaluation: ROC-AUC, Precision-Recall, confusion matrix,
# risk-based threshold simulation, and live loan scoring.
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report,
    confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import THRESHOLD_CONFIG, FEATURE_CONFIG
from src.utils.helpers import print_section, save_report, align_columns
from src.features.feature_engineering import run_feature_engineering


def threshold_simulation(y_test: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Simulate classification metrics and business profit across all thresholds.

    Returns
    -------
    pd.DataFrame with columns: threshold, precision, recall, f1,
                                approval_rate, simulated_profit
    """
    cfg  = THRESHOLD_CONFIG
    gain = cfg['profit_approve_good']
    loss = cfg['loss_approve_bad']

    thresholds = np.arange(cfg['start'], cfg['stop'], cfg['step'])
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        # Business metric: approved good = gain, approved bad = loss
        profit = tp * loss + fp * gain + tn * gain + fn * 0

        rows.append({
            'threshold'       : round(t, 2),
            'precision'       : round(prec, 3),
            'recall'          : round(rec, 3),
            'f1'              : round(f1, 3),
            'approval_rate'   : round(1 - y_pred.mean(), 3),
            'simulated_profit': int(profit),
        })

    df = pd.DataFrame(rows)
    return df


def get_optimal_thresholds(thresh_df: pd.DataFrame) -> tuple:
    """
    Find the optimal threshold by F1 score and by simulated profit.

    Returns
    -------
    (optimal_f1_threshold, optimal_profit_threshold)
    """
    t_f1     = thresh_df.loc[thresh_df['f1'].idxmax(), 'threshold']
    t_profit = thresh_df.loc[thresh_df['simulated_profit'].idxmax(), 'threshold']
    return float(t_f1), float(t_profit)


def print_classification_report(y_test: pd.Series,
                                  y_prob: np.ndarray,
                                  threshold: float):
    """Print full classification report at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n  Classification Report @ threshold = {threshold}")
    print(classification_report(y_test, y_pred,
                                 target_names=['No Default', 'Default']))


def run_full_evaluation(best_name: str,
                         best_result: dict,
                         y_test: pd.Series) -> tuple:
    """
    Full evaluation pipeline for the best model:
    threshold simulation, optimal threshold selection, classification report.

    Returns
    -------
    (thresh_df, optimal_profit_threshold)
    """
    print_section(f"EVALUATION — {best_name}")
    y_prob = best_result['y_prob']

    thresh_df = threshold_simulation(y_test, y_prob)

    print("\n  Threshold Simulation Results:")
    print(thresh_df.to_string(index=False))
    save_report(thresh_df, "threshold_simulation.csv")

    t_f1, t_profit = get_optimal_thresholds(thresh_df)
    print(f"\n  → Optimal Threshold (F1)    : {t_f1}")
    print(f"  → Optimal Threshold (Profit): {t_profit}")

    print_classification_report(y_test, y_prob, t_profit)

    return thresh_df, t_profit


def score_applicant(raw_record: dict,
                     model,
                     feature_cols: list,
                     threshold: float) -> dict:
    """
    Score a single loan applicant and return default probability,
    risk band, and loan decision.

    Parameters
    ----------
    raw_record   : dict of raw applicant features (no engineered features)
    model        : Trained sklearn classifier
    feature_cols : List of feature column names from training
    threshold    : Decision threshold for APPROVE/REJECT

    Returns
    -------
    dict with default_probability, risk_band, decision
    """
    sample = pd.DataFrame([raw_record])
    sample = run_feature_engineering(sample, encode=True)
    sample = align_columns(sample, feature_cols)

    prob = model.predict_proba(sample)[0, 1]

    if prob < 0.20:
        band = 'LOW RISK'
    elif prob < 0.40:
        band = 'MEDIUM RISK'
    elif prob < 0.60:
        band = 'HIGH RISK'
    else:
        band = 'VERY HIGH RISK'

    decision = 'REJECT' if prob >= threshold else 'APPROVE'

    return {
        'default_probability': round(prob, 4),
        'risk_band'          : band,
        'decision'           : decision,
        'threshold_used'     : threshold,
    }
