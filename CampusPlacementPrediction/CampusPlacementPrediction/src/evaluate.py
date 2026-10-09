"""Module 3 - metrics and charts used to compare the algorithms."""
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import config

COLORS = {"Logistic Regression": "#1B2A57", "Decision Tree": "#E08A00", "Random Forest": "#0E8C85"}


def compute_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "F1-score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob)}, pred, prob


def _save(fig, name):
    config.REPORT_DIR.mkdir(exist_ok=True)
    fig.savefig(config.REPORT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    counts = df[config.TARGET].value_counts().sort_index()
    ax[0].bar(["Not placed", "Placed"], counts.values, color=["#C62F4B", "#0E8C85"])
    ax[0].set_title("Class balance")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 8, str(v), ha="center")
    ax[1].boxplot([df[df.placed == 0].cgpa, df[df.placed == 1].cgpa], tick_labels=["Not placed", "Placed"])
    ax[1].set_title("CGPA vs placement")
    rate = df.groupby("internships")[config.TARGET].mean() * 100
    ax[2].bar(rate.index.astype(str), rate.values, color="#1B2A57")
    ax[2].set_title("Placement rate (%) by internships")
    ax[2].set_xlabel("Internships")
    fig.tight_layout()
    _save(fig, "eda.png")


def plot_comparison(results):
    df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(df.columns))
    w = 0.26
    for i, name in enumerate(df.index):
        bars = ax.bar(x + (i - 1) * w, df.loc[name].values, w, label=name, color=COLORS[name])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_confusion(preds, y_test):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, (name, pred) in zip(ax, preds.items()):
        cm = confusion_matrix(y_test, pred)
        a.imshow(cm, cmap="Blues")
        a.set_title(name, fontsize=10)
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_xticklabels(["Not placed", "Placed"], fontsize=8)
        a.set_yticklabels(["Not placed", "Placed"], fontsize=8)
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def plot_roc(probs, y_test):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, prob):.2f})", color=COLORS[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "roc_curves.png")


def plot_importance(rf_pipeline):
    names = rf_pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__")[1] for n in names]
    imp = pd.Series(rf_pipeline.named_steps["clf"].feature_importances_, index=names).sort_values()
    imp = imp.tail(10)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(imp.index, imp.values, color="#0E8C85")
    ax.set_title("Top 10 features (Random Forest)")
    fig.tight_layout()
    _save(fig, "feature_importance.png")
    return imp.sort_values(ascending=False)
