"""
Tea Leaf Disease Detection - Prediction Script
Run the trained model on a single leaf image.
Usage: python src/predict.py --image path/to/leaf.jpg
Author: Jayanth
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

CLASSES = [
    "Healthy",
    "Anthracnose",
    "Algal Leaf",
    "Bird Eye Spot",
    "Brown Blight",
    "Gray Light",
    "Red Leaf Spot",
    "White Spot"
]

# color for each class (used in the visualization)
CLASS_COLORS = {
    "Healthy": "#27ae60",
    "Anthracnose": "#e74c3c",
    "Algal Leaf": "#2980b9",
    "Bird Eye Spot": "#f39c12",
    "Brown Blight": "#8e44ad",
    "Gray Light": "#7f8c8d",
    "Red Leaf Spot": "#c0392b",
    "White Spot": "#bdc3c7"
}


def load_model(model_path="models/tea_disease_model.h5"):
    """Load the trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first with: python src/train.py")
    return tf.keras.models.load_model(model_path)


def preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess a single image for prediction."""
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img) / 255.0   # normalize to [0, 1]
    return np.expand_dims(arr, axis=0), img


def predict(model, img_array):
    """Run prediction and return the class label and confidence."""
    probs = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(probs)
    return CLASSES[idx], probs[idx], probs


def visualize(img_path, label, confidence, all_probs, save_path=None):
    """Show the image alongside a bar chart of class probabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # show the input image
    img = plt.imread(img_path)
    ax1.imshow(img)
    ax1.set_title(f"Prediction: {label}\nConfidence: {confidence*100:.1f}%",
                  fontsize=14, fontweight="bold", color="white", pad=12)
    ax1.axis("off")
    color = CLASS_COLORS[label]
    for spine in ax1.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    # bar chart of probabilities
    colors = [CLASS_COLORS[c] for c in CLASSES]
    bars = ax2.barh(CLASSES, all_probs * 100, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Confidence (%)", color="white")
    ax2.set_title("Class Probabilities", color="white", fontsize=13)
    ax2.set_xlim(0, 100)
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#16213e")

    # add percentage labels on bars
    for bar, prob in zip(bars, all_probs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{prob*100:.1f}%", va="center", color="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Tea Leaf Disease Predictor")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", default="models/tea_disease_model.h5", help="Model path")
    parser.add_argument("--save", default=None, help="Save output image path")
    args = parser.parse_args()

    print("\nTea Leaf Disease Detection")
    print("-" * 40)

    model = load_model(args.model)
    img_array, _ = preprocess_image(args.image)
    label, confidence, all_probs = predict(model, img_array)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence*100:.2f}%")
    print("\nAll class probabilities:")
    for cls, prob in sorted(zip(CLASSES, all_probs), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        print(f"  {cls:<18} {bar} {prob*100:.1f}%")

    visualize(args.image, label, confidence, all_probs, save_path=args.save)


if __name__ == "__main__":
    main()
