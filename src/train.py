"""
Tea Leaf Disease Detection - Training Script
Uses MobileNetV2 with transfer learning to classify tea leaf diseases.
Author: Jayanth
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

# --- Config ---
IMG_SIZE    = (224, 224)   # MobileNetV2 default input size
BATCH_SIZE  = 32
EPOCHS      = 30           # phase 1 epochs
FINE_TUNE_EPOCHS = 10      # phase 2 (fine-tuning) epochs
LEARNING_RATE    = 1e-3
DATA_DIR    = "data/"
MODEL_DIR   = "models/"
RESULTS_DIR = "results/"

CLASSES = [
    "healthy",
    "anthracnose",
    "algal_leaf",
    "bird_eye_spot",
    "brown_blight",
    "gray_light",
    "red_leaf_spot",
    "white_spot"
]

# --- Data loading and augmentation ---
def build_data_generators(data_dir):
    """
    Creates train, validation, and test data generators.
    I'm using heavy augmentation on training data to prevent overfitting
    since the dataset isn't that large.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2   # 80-20 split within training data
    )

    # no augmentation for validation, just rescale
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    # test set - no augmentation at all
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# --- Model architecture ---
def build_model(num_classes):
    """
    Build the model using MobileNetV2 as the base.
    I chose MobileNetV2 because it's lightweight and still gives good accuracy.
    The base is frozen initially - we only train our custom classifier head first.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,       # remove the original classification head
        weights="imagenet"       # use pretrained weights
    )
    base_model.trainable = False  # freeze base model layers

    # build our classifier on top
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)              # dropout to reduce overfitting
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def unfreeze_top_layers(base_model, num_layers=30):
    """
    Unfreeze the top N layers for fine-tuning.
    This lets the model adapt the pretrained features to our specific dataset.
    I found 30 layers works well - too many and it overfits.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    print(f"Fine-tuning top {num_layers} layers of MobileNetV2")


# --- Callbacks ---
def get_callbacks(model_path):
    """Set up callbacks for training."""
    return [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1)
    ]


# --- Plot training history ---
def plot_history(history, phase="phase1", save_dir=RESULTS_DIR):
    """Save accuracy and loss plots for a training phase."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # accuracy plot
    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title(f"Accuracy - {phase}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # loss plot
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title(f"Loss - {phase}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"history_{phase}.png"), dpi=150)
    plt.close()


# --- Main training loop ---
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Tea Leaf Disease Detection - Training")
    print("=" * 60)

    # load data
    train_gen, val_gen, test_gen = build_data_generators(DATA_DIR)
    num_classes = len(train_gen.class_indices)
    print(f"\nClasses detected: {train_gen.class_indices}")

    # build model
    model, base_model = build_model(num_classes)
    model.summary()

    # --- Phase 1: Train only the classifier head ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[Phase 1] Training classifier head (base model frozen)...")
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(os.path.join(MODEL_DIR, "best_phase1.h5"))
    )
    plot_history(h1, phase="phase1")

    # --- Phase 2: Fine-tune top layers of MobileNetV2 ---
    print("\n[Phase 2] Fine-tuning top layers...")
    unfreeze_top_layers(base_model, num_layers=30)

    # use a lower learning rate for fine-tuning to avoid destroying pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(os.path.join(MODEL_DIR, "best_final.h5"))
    )
    plot_history(h2, phase="phase2_finetune")

    # --- Evaluate on test set ---
    print("\n[Evaluation] Testing on unseen data...")
    loss, acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    # save metrics to json
    metrics = {"test_accuracy": round(acc, 4), "test_loss": round(loss, 4)}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save final model
    model.save(os.path.join(MODEL_DIR, "tea_disease_model.h5"))
    print(f"\nModel saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
