import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import Model

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # EfficientNetB0 is a nested model, its output is
        # the final feature maps (7x7x1280)
        if isinstance(layer, tf.keras.Model):
            return layer.name
    raise ValueError("No suitable layer found in model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Get the EfficientNetB0 sub-model directly
    base_model = model.get_layer(last_conv_layer_name)

    with tf.GradientTape() as tape:
        # Run the base model — its output IS the feature maps (7x7x1280)
        conv_outputs = base_model(img_array, training=False)
        tape.watch(conv_outputs)

        # Manually replay the remaining "head" layers on top
        x = conv_outputs
        for layer in model.layers:
            if layer is base_model or isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)

        predictions = x
        loss = predictions[:, 0]

    # Compute gradients of the prediction w.r.t. the feature maps
    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    # Resize heatmap to match original image size
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1],
        original_img.shape[0])
    )

    # Convert heatmap to 0-255 range
    heatmap = np.uint8(255 * heatmap)

    #Apply color map
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original image to 0-255 range if needed
    if original_img.max() <= 1.0:
        original_img = np.uint8(255 * original_img)

    # Blend heatmap with original image
    overlayed = cv2.addWeighted(
        original_img, 1 - alpha,
        heatmap_colored, alpha,
        0
    )

    return overlayed, heatmap_colored
