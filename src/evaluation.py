"""
Model Evaluation Module.

Provides functions for:
- Per-model metric calculation (Accuracy, Precision, Recall, F1)
- Classification report generation
- Confusion matrix visualization
- Multi-model comparison table and bar chart
- Training history plots for deep learning models
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from src.utils import print_header, print_info, print_success, print_subheader, print_metric


def evaluate_model(model_name, y_true, y_pred):
    """Calculate all metrics for a single model.

    Returns:
        dict with Model, Accuracy, Precision, Recall, F1 Score
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  --- {model_name} ---")
    print_metric("Accuracy", f"{acc:.4f}")
    print_metric("Precision", f"{prec:.4f}")
    print_metric("Recall", f"{rec:.4f}")
    print_metric("F1 Score", f"{f1:.4f}")

    print(f"\n  Classification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=['Negative', 'Positive'],
        zero_division=0,
    )
    for line in report.split('\n'):
        print(f"    {line}")

    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax, annot_kws={'size': 14},
        linewidths=0.5, linecolor='white',
    )
    ax.set_title(f'{model_name}\nConfusion Matrix',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_all_models(models_dict, output_dir):
    """Evaluate all models and produce comparison table + chart.

    Args:
        models_dict: dict mapping model_name -> (y_true, y_pred)
        output_dir: path to save outputs

    Returns:
        DataFrame with comparison results (sorted by F1 Score descending).
    """
    print_header("STEP 6: MODEL EVALUATION")

    results = []
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for name, (y_true, y_pred) in models_dict.items():
        result = evaluate_model(name, y_true, y_pred)
        results.append(result)

        # Confusion matrix
        safe_name = name.lower().replace(' ', '_').replace('+', '_')
        cm_path = os.path.join(plots_dir, f'cm_{safe_name}.png')
        plot_confusion_matrix(y_true, y_pred, name, cm_path)

    # ─── Comparison Table ────────────────────────────────
    print_subheader("FINAL MODEL COMPARISON TABLE")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

    # Pretty-print the table
    print()
    print(results_df.to_string(index=False))
    print()

    # Save CSV
    csv_path = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(csv_path, index=False)
    print_success(f"Results saved to {csv_path}")

    # ─── Comparison Bar Chart ────────────────────────────
    _plot_comparison_chart(results_df, plots_dir)

    return results_df


def _plot_comparison_chart(results_df, plots_dir):
    """Create a grouped bar chart comparing all models on all metrics."""
    print_info("Generating model comparison chart...")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    models = results_df['Model'].tolist()
    n_models = len(models)

    x = np.arange(n_models)
    width = 0.18
    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']

    fig, ax = plt.subplots(figsize=(max(12, n_models * 2.5), 7))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = results_df[metric].tolist()
        bars = ax.bar(x + i * width, values, width,
                      label=metric, color=color, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('Models', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Performance Comparison — All Metrics',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0.5, 1.08)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: model_comparison.png")


def plot_training_history(history, model_name, plots_dir):
    """Plot accuracy and loss curves for a deep learning model's training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train',
                 linewidth=2, color='#2ECC71', marker='o', markersize=5)
    axes[0].plot(history.history['val_accuracy'], label='Validation',
                 linewidth=2, color='#E74C3C', marker='s', markersize=5)
    axes[0].set_title(f'{model_name} — Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Loss
    axes[1].plot(history.history['loss'], label='Train',
                 linewidth=2, color='#2ECC71', marker='o', markersize=5)
    axes[1].plot(history.history['val_loss'], label='Validation',
                 linewidth=2, color='#E74C3C', marker='s', markersize=5)
    axes[1].set_title(f'{model_name} — Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    path = os.path.join(plots_dir, f'training_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"Saved: training_{safe_name}.png")
