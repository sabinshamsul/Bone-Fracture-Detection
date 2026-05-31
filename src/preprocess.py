import cv2
import numpy as np
import os
import pandas as pd

from pathlib import Path

def apply_clahe(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the grayscale image
    enhanced = clahe.apply(gray)

    # Convert the enhanced grayscale image back to BGR format
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr

def load_image(image_path, target_size=(224, 224)):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: Could not load image at {image_path}")
        return None

    # Apply CLAHE to enhance the image
    image = apply_clahe(image)

    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0 

    return image

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "dataset.csv"
    images_dir = data_dir / "images"

    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Total records in CSV: {len(df)}")

    images = []
    labels = []
    skipped = 0

    for _, row in df.iterrows():
    # Build the full image path
        if row["fractured"] == 1:
            img_path = images_dir / "Fractured" / row["image_id"]
        else:
            img_path = images_dir / "Non_fractured" / row["image_id"]

        # Load and preprocess the image
        image = load_image(img_path)

        # Only add if image loaded successfully
        if image is not None:
            images.append(image)
            labels.append(row["fractured"])
        else:
            skipped += 1

    print(f"Successfully loaded: {len(images)} images")
    print(f"Skipped (corrupted/missing): {skipped} images")

    #Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Fractured: {int(y.sum())} | Non-fractured: {int(len(y) - y.sum())}")

    return X, y