"""
Tea Leaf Disease Detection - Evaluation Script
Generates confusion matrix, classification report, ROC curves, and per-class metrics.
Author: Jayanth
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = [
    "Healthy", "Anthracnose", "Algal Leaf", "Bird Eye Spot",
    "Brown Blight", "Gray Light", "Red Leaf Spot", "White Spot"
]
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
DATA_DIR   = "data/"
MODEL_PATH = "models/tea_disease_model.h5"
RESULTS    = "results/"


def load_test_data():
    """Load the test dataset."""
    gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return gen


def get_predictions(model, gen):
    """Run the model on the test set and get predictions."""
    y_prob = model.predict(gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = gen.classes
    return y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot and save the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    # convert to percentages so it's easier to read
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of True Class"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (%)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(y_true, y_prob, save_dir):
    """Plot ROC curve for each class."""
    n_classes = len(CLASSES)
    y_true_bin = np.eye(n_classes)[y_true]  # one-hot encode true labels

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    # diagonal line for reference (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Per Class", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_class_metrics(report_dict, save_dir):
    """Bar chart comparing precision, recall, and F1 for each class."""
    metrics = ["precision", "recall", "f1-score"]
    classes = [c for c in CLASSES if c in report_dict]
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict[c][metric] for c in classes]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(RESULTS, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    gen = load_test_data()

    print("Running predictions...")
    y_true, y_pred, y_prob = get_predictions(model, gen)

    # print classification report to console
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # save report as json
    with open(os.path.join(RESULTS, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # generate all the plots
    plot_confusion_matrix(y_true, y_pred, RESULTS)
    plot_roc_curves(y_true, y_prob, RESULTS)
    plot_per_class_metrics(report, RESULTS)

    print(f"\nOverall Accuracy: {report['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {report['macro avg']['f1-score']*100:.2f}%")
    print(f"\nAll results saved to: {RESULTS}")


if __name__ == "__main__":
    main()
