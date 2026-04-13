from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_output_dir(output_dir: Path) -> None:
    # Create the output directory if it does not already exist so every plot
    # save call below can write safely into `results/`.
    output_dir.mkdir(parents=True, exist_ok=True)


def _clean_feature_name(name: str) -> str:
    # Simplify sklearn-generated feature names so the labels are easier to read
    # in the feature-importance chart.
    cleaned = name
    for prefix in ("cat__", "num__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def plot_model_performance_comparison(log_results, rf_results, output_dir: Path) -> Path:
    # Create a side-by-side bar chart for the main hold-out test metrics so the
    # two models can be compared visually at a glance.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "model_performance_comparison.svg"

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    log_values = [log_results[metric] for metric in metrics]
    rf_values = [rf_results[metric] for metric in metrics]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    log_bars = ax.bar(x - width / 2, log_values, width, label="Logistic Regression", color="#2E8B57")
    rf_bars = ax.bar(x + width / 2, rf_values, width, label="Random Forest", color="#C0392B")

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, min(1.1, max(log_values + rf_values) + 0.12))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (log_bars, rf_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve_analysis(log_results, rf_results, output_dir: Path) -> Path:
    # Plot both ROC curves on the same axes so the separation ability of the
    # two models can be compared directly.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "roc_curve_analysis.svg"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        log_results["fpr"],
        log_results["tpr"],
        label=f"Logistic Regression (AUC = {log_results['roc_auc']:.3f})",
        color="#2E8B57",
        linewidth=2,
    )
    ax.plot(
        rf_results["fpr"],
        rf_results["tpr"],
        label=f"Random Forest (AUC = {rf_results['roc_auc']:.3f})",
        color="#C0392B",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Baseline")

    ax.set_title("ROC Curve Analysis")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cv_roc_auc(log_cv, rf_cv, output_dir: Path) -> Path:
    # Plot the ROC-AUC score from each cross-validation fold so the stability
    # of each model is visible across repeated splits.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "five_fold_cv_roc_auc.svg"

    folds = np.arange(1, len(log_cv["scores"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(folds, log_cv["scores"], marker="o", linewidth=2, color="#2E8B57", label="Logistic Regression")
    ax.plot(folds, rf_cv["scores"], marker="o", linewidth=2, color="#C0392B", label="Random Forest")
    ax.axhline(log_cv["mean"], color="#2E8B57", linestyle="--", alpha=0.5)
    ax.axhline(rf_cv["mean"], color="#C0392B", linestyle="--", alpha=0.5)

    ax.set_title("5-Fold Cross-Validation ROC-AUC")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(folds)
    ax.set_ylim(0.5, min(1.05, max(log_cv["scores"].max(), rf_cv["scores"].max()) + 0.08))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    for x_pos, score in zip(folds, log_cv["scores"]):
        ax.text(x_pos, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=9, color="#2E8B57")
    for x_pos, score in zip(folds, rf_cv["scores"]):
        ax.text(x_pos, score - 0.025, f"{score:.3f}", ha="center", va="top", fontsize=9, color="#C0392B")

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_threshold_vs_metrics(log_threshold, rf_threshold, output_dir: Path) -> Path:
    # Plot precision, recall, and F1-score against threshold so the trade-off
    # between aggressive and conservative probability cutoffs is easy to read.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "vs_threshold_analysis.svg"

    thresholds = [row["threshold"] for row in log_threshold]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    model_rows = [
        ("Logistic Regression", log_threshold, "#2E8B57", axes[0]),
        ("Random Forest", rf_threshold, "#C0392B", axes[1]),
    ]

    for title, rows, color, ax in model_rows:
        ax.plot(thresholds, [row["precision"] for row in rows], marker="o", label="Precision", color=color)
        ax.plot(thresholds, [row["recall"] for row in rows], marker="s", label="Recall", color="#1F4E79")
        ax.plot(thresholds, [row["f1"] for row in rows], marker="^", label="F1-Score", color="#B8860B")

        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()

    axes[0].set_ylabel("Score")
    fig.suptitle("Metrics vs Threshold")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_random_forest_feature_importance(model, output_dir: Path, top_n: int = 12) -> Path:
    # Plot the top transformed feature importances from the trained random
    # forest pipeline.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "random_forest_top_12_feature_importance.svg"

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    importances = classifier.feature_importances_

    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[index] for index in top_indices]
    top_values = [importances[index] for index in top_indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_names[::-1], top_values[::-1], color="#C0392B")

    ax.set_title("Random Forest Top 12 Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, max(top_values) + max(top_values) * 0.2)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(top_values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix_analysis(log_results, rf_results, output_dir: Path) -> Path:
    # Plot the confusion matrices for both models side by side so the user can
    # compare true negatives, false positives, false negatives, and true
    # positives visually.
    _ensure_output_dir(output_dir)
    output_path = output_dir / "confusion_matrix_analysis.svg"

    matrices = [
        ("Logistic Regression", log_results["confusion_matrix"], "#2E8B57"),
        ("Random Forest", rf_results["confusion_matrix"], "#C0392B"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, matrix, color) in zip(axes, matrices):
        image = ax.imshow(matrix, cmap="Greens" if color == "#2E8B57" else "Reds")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Retained", "Churned"], rotation=20)
        ax.set_yticklabels(["Retained", "Churned"])

        max_value = matrix.max()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value}",
                    ha="center",
                    va="center",
                    color="white" if value > max_value / 2 else "black",
                    fontsize=11,
                    fontweight="bold",
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrix Analysis")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path
