import tensorflow as tf
import numpy as np
import cv2
import argparse

from pathlib import Path
from src.gradcam import get_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

def load_model_and_setup(model_path="models/bone_fracture_model.keras"):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    last_conv_layer = get_last_conv_layer(model)
    print(f"Model loaded | Last conv layer: {last_conv_layer}")
    return model, last_conv_layer

def predict_and_visualize(image_path, model, 
                          last_conv_layer,
                          output_dir="results/predictions"):
    image_path = Path(image_path)

    #Load and process image
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    original_img = cv2.resize(original_img, (224, 224))

    img_array = original_img.astype(np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    #Get prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Fractured" if prediction > 0.5 else "Non-fractured"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    #Generate Grad-CAM overlay
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    overlayed, _ = overlay_heatmap(heatmap, original_img)

    # Save the result
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_gradcam.jpg"
    cv2.imwrite(str(output_path), overlayed)

    return {
        "image": image_path.name,
        "label": label,
        "confidence": confidence,
        "raw_score": prediction,
        "output_path": output_path
    }

def main():
    parser = argparse.ArgumentParser(
        description="Predict bone fractures from X-ray images with Grad-CAM"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to X-ray image(s) to analyse"
    )
    args = parser.parse_args()

    model, last_conv_layer = load_model_and_setup()

    print("\n" + "=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    results = []
    for image_path in args.images:
        result = predict_and_visualize(image_path, model, last_conv_layer)
        if result: 
            results.append(result)
            print(f"\n {result['image']}")
            print(f" Prediction: {result['label']} "
                  f"({result['confidence']:.1%} confidence)")
            print(f" Grad-CAM saved: {result['output_path']}")

    print(f"\n" + "=" * 60)
    print(f" Done! Processed {len(results)}/{len(args.images)} images")
    print("=" * 60)

if __name__ == "__main__":
    main()