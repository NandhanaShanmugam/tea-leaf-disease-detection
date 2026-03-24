"""
Tea Leaf Disease Detection - Dataset Preparation
Downloads the dataset from Kaggle and splits it into train/val/test folders.
Author: Jayanth
"""

import os
import shutil
import zipfile
import argparse
import random
from pathlib import Path

# these are the 8 disease classes in our dataset
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


def download_kaggle_dataset(dataset_slug="shashwatwork/tea-leaf-disease-dataset", dest="data/raw"):
    """Download the dataset using the Kaggle API."""
    os.makedirs(dest, exist_ok=True)
    try:
        import kaggle
        print(f"Downloading {dataset_slug} ...")
        kaggle.api.dataset_download_files(dataset_slug, path=dest, unzip=True)
        print("Download complete!")
    except ImportError:
        print("Kaggle not installed. Run: pip install kaggle")
        print("Then set up your kaggle.json credentials (see https://www.kaggle.com/docs/api)")
        raise
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def organize_dataset(raw_dir="data/raw", out_dir="data", val_split=0.15, test_split=0.15, seed=42):
    """
    Split the raw images into train/val/test directories.
    Each class gets its own subfolder in each split.
    Default split: 70% train, 15% val, 15% test
    """
    random.seed(seed)

    # create all the output directories
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            Path(os.path.join(out_dir, split, cls)).mkdir(parents=True, exist_ok=True)

    total = 0
    stats = {}

    for cls in CLASSES:
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_path):
            print(f"Warning: Class folder not found: {cls_path}")
            continue

        # get all image files
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(imgs)

        # calculate split sizes
        n = len(imgs)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)
        n_train = n - n_test - n_val

        splits_imgs = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train + n_val],
            "test":  imgs[n_train + n_val:]
        }

        # copy files to the right directories
        for split, files in splits_imgs.items():
            for f in files:
                src = os.path.join(cls_path, f)
                dst = os.path.join(out_dir, split, cls, f)
                shutil.copy2(src, dst)

        stats[cls] = {"train": n_train, "val": n_val, "test": n_test}
        total += n
        print(f"  {cls:<20} train={n_train:>4}  val={n_val:>3}  test={n_test:>3}  total={n}")

    print(f"\nDataset organized! Total images: {total}")
    return stats


def verify_dataset(data_dir="data"):
    """Quick check to make sure the dataset was split correctly."""
    print("\nDataset structure:")
    for split in ["train", "val", "test"]:
        print(f"\n  {split}/")
        total = 0
        for cls in CLASSES:
            path = os.path.join(data_dir, split, cls)
            if os.path.exists(path):
                n = len(os.listdir(path))
                total += n
                print(f"    {cls:<20} {n:>4} images")
        print(f"    {'TOTAL':<20} {total:>4} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the tea leaf disease dataset")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw downloaded data")
    parser.add_argument("--out_dir", default="data", help="Output directory for splits")
    parser.add_argument("--verify",  action="store_true", help="Print dataset summary")
    args = parser.parse_args()

    if args.download:
        download_kaggle_dataset(dest=args.raw_dir)

    organize_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir)

    if args.verify:
        verify_dataset(args.out_dir)
