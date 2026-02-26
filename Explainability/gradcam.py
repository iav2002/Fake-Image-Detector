"""
Grad-CAM Visualizations for Fine-Tuned ResNet50
Generates heatmaps showing what regions the model focuses on
when classifying images as real vs AI-generated.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys


# ---- CONFIG ----
MODEL_PATH = "/Users/ignacioalarconvarela/Developer/GRAD-CAM/my_ai_detector_resnet50.keras"  # <-- UPDATE THIS
IMG_DIR = "/Users/ignacioalarconvarela/Developer/GRAD-CAM/images"           # <-- UPDATE THIS
OUTPUT_DIR = "gradcam_outputs"
IMG_SIZE = (224, 224)

# Last conv layer in ResNet50 — this is standard
# Run with --layers flag to verify if needed
LAST_CONV_LAYER = "conv5_block3_out"


def find_conv_layers(model):
    """Print all conv layer names to help identify the right target layer."""
    print("\n--- Convolutional layers in model ---")
    for layer in model.layers:
        if "conv" in layer.name.lower():
            try:
                shape = layer.output.shape
            except Exception:
                shape = "unknown"
            print(f"  {layer.name} -> output shape: {shape}")
    print("---\n")


def load_and_preprocess(img_path):
    """Load image and apply ResNet50 preprocessing."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    processed = tf.keras.applications.resnet50.preprocess_input(arr.copy())
    return img, processed


def generate_gradcam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap for a given image."""
    # Build a model that outputs both the conv layer and the final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    # Compute gradients of the prediction w.r.t. the conv layer output
    with tf.GradientTape() as tape:
        results = grad_model(img_array)
        # Keras 3 can nest outputs — flatten to tensors
        conv_output = tf.convert_to_tensor(results[0])
        tape.watch(conv_output)
        predictions = tf.convert_to_tensor(results[1])
        pred_value = tf.reshape(predictions, [-1])

    grads = tape.gradient(pred_value, conv_output)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by gradient importance
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    # Extract scalar prediction
    pred_np = float(pred_value.numpy()[0])
    return heatmap.numpy(), pred_np


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Overlay the heatmap on the original image."""
    # Resize heatmap to match image
    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    # Resize to original image dimensions
    jet_heatmap = tf.image.resize(
        np.expand_dims(jet_heatmap, 0),
        (original_img.size[1], original_img.size[0])
    ).numpy()[0]

    # Combine
    original_arr = np.array(original_img) / 255.0
    overlay = jet_heatmap * alpha + original_arr * (1 - alpha)
    return np.clip(overlay, 0, 1)


def process_images(model, img_dir, output_dir):
    """Process all images in directory and save Grad-CAM outputs."""
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    img_files = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    if not img_files:
        print(f"No images found in {img_dir}")
        return

    print(f"Processing {len(img_files)} images...\n")

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        original, processed = load_and_preprocess(img_path)
        heatmap, pred_score = generate_gradcam(model, processed, LAST_CONV_LAYER)
        overlay = overlay_heatmap(original, heatmap)

        # Prediction label
        label = "Real" if pred_score > 0.5 else "AI-Generated"
        confidence = pred_score if pred_score > 0.5 else 1 - pred_score

        # Save figure with original + heatmap + overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Prediction: {label} ({confidence:.1%})", fontsize=12)
        axes[2].axis("off")

        fig.suptitle(fname, fontsize=10, color="gray")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"gradcam_{fname}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  {fname} -> {label} ({confidence:.1%}) | Saved: {out_path}")

    print(f"\nDone. All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    # Optional: just list conv layers to find the right one
    if "--layers" in sys.argv:
        model = load_model(MODEL_PATH)
        find_conv_layers(model)
        sys.exit(0)

    # Validate paths
    if "path/to" in MODEL_PATH or "path/to" in IMG_DIR:
        print("ERROR: Update MODEL_PATH and IMG_DIR in the script before running.")
        sys.exit(1)

    print("Loading model...")
    model = load_model(MODEL_PATH)
    print(f"Model loaded. Target conv layer: {LAST_CONV_LAYER}\n")

    process_images(model, IMG_DIR, OUTPUT_DIR)
