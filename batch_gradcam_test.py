# batch_gradcam_test.py
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path

from src.gradcam import get_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

model = tf.keras.models.load_model("models/bone_fracture_model.keras")
last_conv_layer = get_last_conv_layer(model)
df = pd.read_csv("data/FracAtlas/dataset.csv")

fractured_paths = list(Path("data/FracAtlas/images/Fractured").glob("*.jpg"))
non_fractured_paths = list(Path("data/FracAtlas/images/Non_fractured").glob("*.jpg"))

# Pick 5 fractured + 5 non-fractured
samples = random.sample(fractured_paths, 5) + random.sample(non_fractured_paths, 5)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, img_path in enumerate(samples):
    original_img = cv2.imread(str(img_path))
    original_img = cv2.resize(original_img, (224, 224))

    img_array = original_img.astype(np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Fractured" if prediction > 0.5 else "Non-fractured"

    row = df[df["image_id"] == img_path.name]
    ground_truth = int(row["fractured"].values[0])
    ground_truth_label = "Fractured" if ground_truth == 1 else "Non-fractured"
    predicted_class = 1 if prediction > 0.5 else 0
    correct = "Correct" if predicted_class == ground_truth else "Incorrect"
    title_color = "green" if predicted_class == ground_truth else "red"

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    overlayed, _ = overlay_heatmap(heatmap, original_img)

    ax = axes[i // 5, i % 5]
    ax.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{label} ({prediction:.1%})\nActual: {ground_truth_label} ({correct})",
                  fontsize=9, color=title_color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/gradcam_batch.png", dpi=150)
plt.show()
print("✅ Saved to results/gradcam_batch.png")