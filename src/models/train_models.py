# src/models/train_models.py
# ─────────────────────────────────────────────────────────────
# Defines, trains, and cross-validates all classifiers.
# Returns trained models + their metrics.
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import MODEL_CONFIG, CV_CONFIG
from src.utils.helpers import print_section


def build_models() -> dict:
    """
    Instantiate all classifiers using config hyperparameters.

    Returns
    -------
    dict of {model_name: sklearn_estimator}
    """
    cfg = MODEL_CONFIG
    return {
        'Logistic Regression': LogisticRegression(**cfg['logistic_regression']),
        'Random Forest'      : RandomForestClassifier(**cfg['random_forest']),
        'Gradient Boosting'  : GradientBoostingClassifier(**cfg['gradient_boosting']),
    }


def cross_validate_model(model, X_train, y_train) -> tuple:
    """
    Run stratified k-fold cross-validation on a model.

    Returns
    -------
    (mean_auc, std_auc)
    """
    cv = StratifiedKFold(
        n_splits=CV_CONFIG['n_splits'],
        shuffle=CV_CONFIG['shuffle'],
        random_state=CV_CONFIG['random_state']
    )
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring=CV_CONFIG['scoring']
    )
    return scores.mean(), scores.std()


def train_and_evaluate(models: dict,
                        X_train, y_train,
                        X_train_sc, y_train_cv,
                        X_test, y_test,
                        X_test_sc) -> dict:
    """
    Train all models and collect performance metrics.

    Logistic Regression uses scaled input; tree models use raw features.

    Parameters
    ----------
    models       : dict of model name → estimator
    X_train      : Raw training features (for tree models)
    y_train      : Training labels
    X_train_sc   : Scaled training features (for LR)
    X_test       : Raw test features
    y_test       : Test labels
    X_test_sc    : Scaled test features

    Returns
    -------
    dict of model_name → {model, y_prob, auc, ap, cv_mean, cv_std}
    """
    print_section("MODEL TRAINING & CROSS-VALIDATION")
    results = {}

    for name, model in models.items():
        is_linear = (name == 'Logistic Regression')
        X_tr = X_train_sc if is_linear else X_train
        X_te = X_test_sc  if is_linear else X_test

        # Cross-validate
        cv_mean, cv_std = cross_validate_model(model, X_tr, y_train)

        # Fit on full training set
        model.fit(X_tr, y_train)

        # Predict probabilities
        y_prob = model.predict_proba(X_te)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)

        results[name] = {
            'model'  : model,
            'y_prob' : y_prob,
            'auc'    : auc,
            'ap'     : ap,
            'cv_mean': cv_mean,
            'cv_std' : cv_std,
            'X_te'   : X_te,
            'X_tr'   : X_tr,
        }

        print(f"\n  {name}")
        print(f"    CV ROC-AUC : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Test AUC   : {auc:.4f}")
        print(f"    Avg Prec.  : {ap:.4f}")

    return results


def get_best_model(results: dict) -> tuple:
    """
    Pick the model with the highest test AUC.

    Returns
    -------
    (best_model_name, best_result_dict)
    """
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  ★ Best Model : {best_name}  (AUC = {results[best_name]['auc']:.4f})")
    return best_name, results[best_name]
