import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.gradcam import get_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

# Step 1: Load the trained model
print("Loading model...")
model = tf.keras.models.load_model("models/bone_fracture_model.keras")
print("Model loaded")

# Step 2: Find the last conv layer 
last_conv_layer = get_last_conv_layer(model)
print(f"Last conv layer: {last_conv_layer}")

# Step 3: Load a RANDOM X-ray (mix of fractured + non-fractured)
import random

fractured_paths = list(Path("data/FracAtlas/images/Fractured").glob("*.jpg"))
non_fractured_paths = list(Path("data/FracAtlas/images/Non_fractured").glob("*.jpg"))

all_paths = fractured_paths + non_fractured_paths
img_path = random.choice(all_paths)
print(f"Using image: {img_path}")

# Step 4: Preprocess the image (same as training) 
original_img = cv2.imread(str(img_path))
original_img = cv2.resize(original_img, (224, 224))

img_array = original_img.astype(np.float32)
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Step 5: Get prediction 
prediction = model.predict(img_array, verbose=0)[0][0]
label = "Fractured" if prediction > 0.5 else "Non-fractured"
print(f"Prediction: {label} ({prediction:.4f})")

# Look up ground truth label from CSV
import pandas as pd
df = pd.read_csv("data/FracAtlas/dataset.csv")
row = df[df["image_id"] == img_path.name]
ground_truth = int(row["fractured"].values[0])
ground_truth_label = "Fractured" if ground_truth == 1 else "Non-fractured"

predicted_class = 1 if prediction > 0.5 else 0
correct = "Correct" if predicted_class == ground_truth else "Incorrect"
print(f"Ground Truth: {ground_truth_label}")
print(f"Match: {correct}")

# Step 6: Generate Grad-CAM heatmap 
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
print(f"Heatmap shape: {heatmap.shape}")

# Step 7: Overlay heatmap on original image 
overlayed, heatmap_colored = overlay_heatmap(heatmap, original_img)

# Step 8: Display results 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original X-ray")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
title_color = "green" if predicted_class == ground_truth else "red"
axes[2].set_title(
    f"Predicted: {label} ({prediction:.2%})\n"
    f"Actual: {ground_truth_label} {correct}",
    color=title_color
) 
axes[2].axis("off")

plt.tight_layout()
plt.savefig("results/gradcam_sample.png", dpi=150)
plt.show()

print("\nSaved to results/gradcam_sample.png")