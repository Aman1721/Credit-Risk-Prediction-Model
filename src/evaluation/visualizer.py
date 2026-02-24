# src/evaluation/visualizer.py
# ─────────────────────────────────────────────────────────────
# All plotting functions: ROC curves, PR curves, feature
# importance, confusion matrix, threshold simulation plots.
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import VIZ_CONFIG, PLOTS_DIR, FEATURE_CONFIG


# ── Shared style setup ────────────────────────────────────────
BG   = VIZ_CONFIG['bg_color']
CARD = VIZ_CONFIG['card_color']
ACC  = VIZ_CONFIG['accent']
NEG  = VIZ_CONFIG['negative']
POS  = VIZ_CONFIG['positive']
WARN = VIZ_CONFIG['warning']
TXT  = VIZ_CONFIG['text']
CLR  = [ACC, NEG, POS]

plt.rcParams.update({
    'text.color'       : TXT,
    'axes.labelcolor'  : TXT,
    'xtick.color'      : TXT,
    'ytick.color'      : TXT,
    'axes.edgecolor'   : '#333355',
    'figure.facecolor' : BG,
})


def _style_ax(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color('#333355')


def plot_class_distribution(y: pd.Series, ax):
    counts = y.value_counts().sort_index()
    bars = ax.bar(['No Default', 'Default'], counts.values,
                   color=[POS, NEG], width=0.5, edgecolor='none')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f'{val}\n({val/len(y):.1%})',
                ha='center', va='bottom', color=TXT, fontsize=10, fontweight='bold')
    ax.set_title('Class Distribution', color=TXT, fontsize=12, pad=10)
    _style_ax(ax)


def plot_feature_distribution(df_raw: pd.DataFrame, col: str,
                                label_col: str, ax, title: str):
    for label, color, name in [(0, POS, 'No Default'), (1, NEG, 'Default')]:
        ax.hist(df_raw[df_raw[label_col] == label][col], bins=40,
                alpha=0.7, color=color, label=name, edgecolor='none')
    ax.set_title(title, color=TXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, labelcolor=TXT)
    _style_ax(ax)


def plot_roc_curves(results: dict, y_test: pd.Series, ax):
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, label='Random Baseline')
    for (name, res), color in zip(results.items(), CLR):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', color=TXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, labelcolor=TXT, loc='lower right')
    _style_ax(ax)


def plot_pr_curves(results: dict, y_test: pd.Series, ax):
    for (name, res), color in zip(results.items(), CLR):
        prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AP={res['ap']:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', color=TXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, labelcolor=TXT, fontsize=8)
    _style_ax(ax)


def plot_feature_importance(model, feature_cols: list, ax, top_n: int = 15):
    imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(top_n)
    ax.barh(imp.index[::-1], imp.values[::-1], color=ACC, edgecolor='none')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)',
                 color=TXT, fontsize=12, pad=10)
    ax.set_xlabel('Importance Score')
    _style_ax(ax)


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray,
                            threshold: float, ax):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                ax=ax, cbar=False, linewidths=1, linecolor=BG)
    ax.set_title(f'Confusion Matrix\n(threshold={threshold})', color=TXT, fontsize=11, pad=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    _style_ax(ax)


def plot_threshold_metrics(thresh_df: pd.DataFrame,
                            t_f1: float, t_profit: float, ax):
    ax2 = ax.twinx()
    ax.plot(thresh_df['threshold'], thresh_df['f1'],
            color=ACC, lw=2, marker='o', ms=4, label='F1 Score')
    ax.plot(thresh_df['threshold'], thresh_df['precision'],
            color=WARN, lw=2, ls='--', label='Precision')
    ax.plot(thresh_df['threshold'], thresh_df['recall'],
            color=POS, lw=2, ls=':', label='Recall')
    ax2.plot(thresh_df['threshold'], thresh_df['approval_rate'],
             color=NEG, lw=2, ls='-.', label='Approval Rate')
    ax.axvline(t_f1, color=ACC, alpha=0.5, ls='--')
    ax.axvline(t_profit, color=NEG, alpha=0.5, ls='--')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score', color=TXT)
    ax2.set_ylabel('Approval Rate', color=NEG)
    ax2.tick_params(colors=NEG)
    ax.set_title('Threshold vs Metrics & Approval Rate', color=TXT, fontsize=12, pad=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              facecolor=CARD, labelcolor=TXT, fontsize=8)
    _style_ax(ax)


def plot_profit_curve(thresh_df: pd.DataFrame, t_profit: float, ax):
    ax.plot(thresh_df['threshold'], thresh_df['simulated_profit'] / 1000,
            color=POS, lw=2.5, marker='D', ms=4)
    ax.axvline(t_profit, color=WARN, ls='--', label=f'Optimal={t_profit}')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Profit ($K)')
    ax.set_title('Simulated Profit by Threshold', color=TXT, fontsize=12, pad=10)
    ax.legend(facecolor=CARD, labelcolor=TXT)
    _style_ax(ax)


def build_dashboard(df_raw: pd.DataFrame,
                     y: pd.Series,
                     results: dict,
                     y_test: pd.Series,
                     thresh_df: pd.DataFrame,
                     t_f1: float,
                     t_profit: float,
                     y_pred_final: np.ndarray,
                     feature_cols: list,
                     rf_model) -> str:
    """
    Compose the full 9-panel dashboard and save to outputs/plots/.

    Returns
    -------
    Path to saved PNG
    """
    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0 — dataset overview
    ax0 = fig.add_subplot(gs[0, 0]); plot_class_distribution(y, ax0)
    ax1 = fig.add_subplot(gs[0, 1])
    plot_feature_distribution(df_raw, 'credit_score', 'default', ax1,
                               'Credit Score by Default Status')
    ax2 = fig.add_subplot(gs[0, 2])
    plot_feature_distribution(df_raw, 'dti_ratio', 'default', ax2,
                               'DTI Ratio by Default Status')

    # Row 1 — model curves
    ax3 = fig.add_subplot(gs[1, 0:2]); plot_roc_curves(results, y_test, ax3)
    ax4 = fig.add_subplot(gs[1, 2]);   plot_pr_curves(results, y_test, ax4)

    # Row 2 — feature importance + confusion matrix
    ax5 = fig.add_subplot(gs[2, 0:2])
    plot_feature_importance(rf_model, feature_cols, ax5)
    ax6 = fig.add_subplot(gs[2, 2])
    plot_confusion_matrix(y_test, y_pred_final, t_profit, ax6)

    # Row 3 — threshold analysis
    ax7 = fig.add_subplot(gs[3, 0:2])
    plot_threshold_metrics(thresh_df, t_f1, t_profit, ax7)
    ax8 = fig.add_subplot(gs[3, 2])
    plot_profit_curve(thresh_df, t_profit, ax8)

    fig.suptitle('Credit Risk Prediction — ML Dashboard',
                 fontsize=20, fontweight='bold', color=TXT, y=0.98)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, 'credit_risk_dashboard.png')
    plt.savefig(path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualizer] Dashboard saved → {path}")
    return path
