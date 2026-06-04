import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf

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

def create_tf_dataset(data_dir, batch_size=32, img_size=(224, 224), 
                      subset="training", seed=42):
    data_dir = Path(data_dir)
    csv_path = data_dir / "dataset.csv"
    images_dir = data_dir / "images"
    
    df = pd.read_csv(csv_path)
    
    # Build full paths and labels
    image_paths = []
    labels = []
    
    for _, row in df.iterrows():
        if row["fractured"] == 1:
            img_path = str(images_dir / "Fractured" / row["image_id"])
        else:
            img_path = str(images_dir / "Non_fractured" / row["image_id"])
        image_paths.append(img_path)
        labels.append(int(row["fractured"]))
    
    # Split indices
    total = len(image_paths)
    indices = list(range(total))
    
    import random
    random.seed(seed)
    random.shuffle(indices)
    
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    
    if subset == "training":
        selected = indices[:train_end]
    elif subset == "validation":
        selected = indices[train_end:val_end]
    else:
        selected = indices[val_end:]
    
    sel_paths = [image_paths[i] for i in selected]
    sel_labels = [labels[i] for i in selected]
    
    print(f"{subset}: {len(sel_paths)} images | "
          f"Fractured: {sum(sel_labels)} | "
          f"Non-fractured: {len(sel_labels)-sum(sel_labels)}")
    
    def parse_image(path, label):
        img = tf.io.read_file(path)
        # decode_image handles corrupted files better than decode_jpeg
        img = tf.image.decode_image(
            img, 
            channels=3, 
            expand_animations=False
        )
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([img_size[0], img_size[1], 3])
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((sel_paths, sel_labels))
    dataset = dataset.map(parse_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if subset == "training":
        dataset = dataset.shuffle(buffer_size=500, seed=seed)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, sel_labels