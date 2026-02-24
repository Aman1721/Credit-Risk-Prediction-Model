# config/config.py
# ─────────────────────────────────────────────────────────────
# Central configuration — all hyperparameters, paths & settings
# ─────────────────────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# ── Dataset ─────────────────────────────────────────────────
DATA_CONFIG = {
    "n_samples"   : 5000,
    "random_seed" : 42,
    "raw_file"    : os.path.join(DATA_DIR, "raw_loans.csv"),
    "test_size"   : 0.2,
}

# ── Feature Engineering ─────────────────────────────────────
FEATURE_CONFIG = {
    "target_col"  : "default",
    "cat_cols"    : ["home_ownership", "loan_purpose", "credit_band", "age_group"],
    "log_cols"    : ["annual_income", "loan_amount"],
    "credit_bins" : [299, 579, 669, 739, 799, 850],
    "credit_labels": ["Poor", "Fair", "Good", "VeryGood", "Exceptional"],
    "age_bins"    : [20, 30, 40, 50, 70],
    "age_labels"  : ["20s", "30s", "40s", "50+"],
}

# ── Model Hyperparameters ───────────────────────────────────
MODEL_CONFIG = {
    "logistic_regression": {
        "max_iter"    : 1000,
        "class_weight": "balanced",
        "C"           : 0.5,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth"   : 8,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs"      : -1,
    },
    "gradient_boosting": {
        "n_estimators" : 200,
        "learning_rate": 0.05,
        "max_depth"    : 4,
        "random_state" : 42,
    },
}

# ── Cross-Validation ─────────────────────────────────────────
CV_CONFIG = {
    "n_splits"  : 5,
    "shuffle"   : True,
    "random_state": 42,
    "scoring"   : "roc_auc",
}

# ── Threshold Simulation ─────────────────────────────────────
THRESHOLD_CONFIG = {
    "start"           : 0.10,
    "stop"            : 0.90,
    "step"            : 0.05,
    "profit_approve_good" : 500,    # $ earned per correctly approved loan
    "loss_approve_bad"    : -2000,  # $ lost per approved defaulter
}

# ── Visualization ─────────────────────────────────────────────
VIZ_CONFIG = {
    "bg_color"   : "#0f1117",
    "card_color" : "#1a1d2e",
    "accent"     : "#6c63ff",
    "negative"   : "#f72585",
    "positive"   : "#2ecc71",
    "warning"    : "#f9c74f",
    "text"       : "#e0e0e0",
    "dpi"        : 150,
}
