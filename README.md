# Tea Leaf Disease Detection

This is my final year project for detecting diseases in tea leaves using deep learning. I used **MobileNetV2** (a pretrained model from ImageNet) and fine-tuned it to classify tea leaf images into 8 categories.

The model achieved around **94% accuracy** on the test set which I'm pretty happy with!

## Why This Project?

Tea is one of the most consumed beverages in the world. Diseases in tea plants can cause huge crop losses for farmers. I wanted to build something that could help identify diseases early from just a photo of the leaf, so farmers can take action quickly.

## Disease Classes

The dataset has 8 categories:

| Class | Description |
|-------|-------------|
| Healthy | No disease present |
| Anthracnose | Fungal infection causing dark lesions |
| Algal Leaf | Green algal growth on leaf surface |
| Bird Eye Spot | Circular spots with dark center |
| Brown Blight | Brown discoloration of leaves |
| Gray Light | Grayish fungal growth |
| Red Leaf Spot | Red/purple circular spots |
| White Spot | White powdery fungal spots |

## How the Model Works

I used transfer learning with MobileNetV2 as the backbone. The training happens in two phases:

1. **Phase 1 (Feature Extraction):** Freeze MobileNetV2 and only train the classifier head (30 epochs)
2. **Phase 2 (Fine-tuning):** Unfreeze the top 30 layers and train at a lower learning rate (10 epochs)

```
Input Image (224x224x3)
        |
  MobileNetV2 (pretrained on ImageNet)
        |
  GlobalAveragePooling2D
  BatchNormalization
  Dense(256) + Dropout(0.4)
  Dense(128) + Dropout(0.3)
  Dense(8, softmax)
        |
  Predicted Class
```

## Project Structure

```
tea-disease-detection/
├── data/
│   ├── raw/           # Raw downloaded dataset
│   ├── train/         # Training images (organized by class)
│   ├── val/           # Validation images
│   └── test/          # Test images
├── models/            # Saved model checkpoints
├── notebooks/
│   └── exploration.ipynb   # Data exploration and visualization
├── results/           # Plots and metrics after evaluation
├── src/
│   ├── prepare_data.py  # Download and organize the dataset
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation and plotting
│   └── predict.py       # Run predictions on new images
├── requirements.txt
├── README.md
└── LICENSE
```

## How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/jayanth/tea-disease-detection.git
cd tea-disease-detection
pip install -r requirements.txt
```

### 2. Get the dataset

You can download it from Kaggle using the API:

```bash
# Make sure you have kaggle.json set up (see https://www.kaggle.com/docs/api)
python src/prepare_data.py --download --verify
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/tea-leaf-disease-dataset), extract to `data/raw/`, then run:

```bash
python src/prepare_data.py --raw_dir data/raw --out_dir data --verify
```

### 3. Train the model

```bash
python src/train.py
```

This will take a while depending on your hardware. The training plots will be saved in `results/`.

### 4. Evaluate

```bash
python src/evaluate.py
```

This generates the confusion matrix, ROC curves, and classification report.

### 5. Predict on a new image

```bash
python src/predict.py --image path/to/your/leaf.jpg
```

You can also save the output visualization:

```bash
python src/predict.py --image path/to/leaf.jpg --save results/prediction.png
```

### 6. Explore the notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Results

- **Test Accuracy:** ~94%
- **Macro F1-Score:** ~93.8%

The confusion matrix and other plots are saved in the `results/` folder after running the evaluation script.

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- scikit-learn (for metrics)
- matplotlib & seaborn (for plots)
- Jupyter Notebook

## License

MIT License — see [LICENSE](LICENSE)

## Author

Jayanth — Final year project, 2025
