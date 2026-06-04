import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

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
    image = cv2.imread(str(image_path))

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

def get_valid_image_paths(data_dir):
    """Pre-scan all images and return only valid ones."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "dataset.csv"
    images_dir = data_dir / "images"
    
    df = pd.read_csv(csv_path)
    
    valid_paths = []
    valid_labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        if row["fractured"] == 1:
            img_path = images_dir / "Fractured" / row["image_id"]
        else:
            img_path = images_dir / "Non_fractured" / row["image_id"]
        
        # Try loading with OpenCV — skip if corrupted
        img = cv2.imread(str(img_path))
        if img is not None and img.size > 0:
            valid_paths.append(str(img_path))
            valid_labels.append(int(row["fractured"]))
        else:
            skipped += 1
    
    print(f"✅ Valid images: {len(valid_paths)}")
    print(f"❌ Skipped corrupted: {skipped}")
    return valid_paths, valid_labels

def create_tf_dataset(image_paths, labels, batch_size=32, 
                      img_size=(224, 224), shuffle=False, seed=42):
    """Create a tf.data.Dataset from pre-validated image paths."""
    
    def parse_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(
            img, 
            channels=3,
            try_recover_truncated=True,  # handles non-standard JPEGs
            acceptable_fraction=0.5      # accepts images even if 50% is missing
        )
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([img_size[0], img_size[1], 3])
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset