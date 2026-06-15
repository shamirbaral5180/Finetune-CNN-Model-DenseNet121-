import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---- CONFIG ----
MODEL_PATH = "outputs_20250906_142440/model_best.h5"
# If you saved your training generator.class_indices as JSON, point to it here:
CLASS_INDEX_JSON = "class_indices.json"   # set to None if you don't have it

# If you trained on DenseNet121 without a Rescaling layer, keep this True.
# If your exported model already includes a Rescaling/Normalization, set to False.
USE_DENSENET_PREPROCESS = True

# Fallback labels if class_indices.json is not available (must match training order!)
FALLBACK_CLASS_LABELS = [
    "Anthracnose",
    "Bacterial Blight",
    "Citrus Canker",
    "Curl Virus",
    "Deficiency Leaf",
    "Dry Leaf",
    "Healthy Leaf",
    "Sooty Mould",
    "Spider Mites"
]

# ---- LOAD MODEL ----
model = load_model(MODEL_PATH)

# Infer expected input size (H, W, C)
try:
    _, H, W, C = model.input_shape
except Exception:
    # Some models use None in the batch dimension
    H, W, C = model.input_shape[1], model.input_shape[2], model.input_shape[3]

if H is None or W is None:
    # Default to 224 if not fixed; change if you trained with a different size
    H = W = 224

# ---- LABELS (ensure correct order!) ----
def load_class_labels():
    if CLASS_INDEX_JSON and os.path.exists(CLASS_INDEX_JSON):
        with open(CLASS_INDEX_JSON, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        # class_indices is typically like {"Anthracnose":0, "Bacterial Blight":1, ...}
        labels = [None] * len(class_indices)
        for k, v in class_indices.items():
            labels[v] = k
        if any(l is None for l in labels):
            raise ValueError("class_indices.json did not produce a complete ordered label list.")
        return labels
    # Fallback to hardcoded list (risky if training order differs)
    return FALLBACK_CLASS_LABELS

CLASS_LABELS = load_class_labels()

# ---- PREPROCESSING ----
# If you trained on DenseNet (Keras applications) WITHOUT an in-model Rescaling layer:
if USE_DENSENET_PREPROCESS:
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
else:
    densenet_preprocess = None

def model_has_rescaling(m):
    """Detect if model already includes a Rescaling/Normalization to avoid double-scaling."""
    for layer in m.layers:
        # Covers both tf.keras.layers.Rescaling and Normalization
        if isinstance(layer, tf.keras.layers.Rescaling) or isinstance(layer, tf.keras.layers.Normalization):
            return True
    return False

HAS_INMODEL_RESCALE = model_has_rescaling(model)

def load_and_preprocess(img_path: str) -> np.ndarray:
    # Load RGB image consistently
    img = image.load_img(img_path, target_size=None, color_mode="rgb")
    arr = image.img_to_array(img)

    # Smart resize with letterbox padding to preserve aspect ratio
    arr = tf.image.resize_with_pad(arr, target_height=H, target_width=W).numpy()

    # Apply preprocessing exactly once:
    if densenet_preprocess and not HAS_INMODEL_RESCALE:
        # Keras DenseNet expects preprocess_input (no manual /255 here)
        arr = densenet_preprocess(arr)
    else:
        # Generic path: scale to [0,1] only if the model doesn't already rescale
        if not HAS_INMODEL_RESCALE:
            arr = arr / 255.0

    # Add batch dimension
    return np.expand_dims(arr, axis=0)

# ---- PREDICTION UTILS ----
def ensure_softmax(pred: np.ndarray) -> np.ndarray:
    """If model outputs logits or unnormalized scores, apply softmax."""
    # If values already ~probabilities summing to 1, leave as-is
    s = pred.sum()
    if 0.99 <= s <= 1.01 and np.all(pred >= 0.0):
        return pred
    # Else apply softmax across classes
    e = np.exp(pred - np.max(pred))
    return e / e.sum()

def predict_image(img_path: str, top_k: int = 3):
    x = load_and_preprocess(img_path)
    raw = model.predict(x, verbose=0)[0]
    probs = ensure_softmax(raw)

    # Top-k
    top_indices = probs.argsort()[-top_k:][::-1]
    top = [(CLASS_LABELS[i], float(probs[i])) for i in top_indices]

    print(f"\nImage: {os.path.basename(img_path)}")
    print("Top predictions:")
    for name, p in top:
        print(f"  {name}: {p*100:.2f}%")

    pred_label, pred_conf = CLASS_LABELS[top_indices[0]], float(probs[top_indices[0]])
    print(f"Final Predicted Class: {pred_label} ({pred_conf*100:.2f}%)")
    print("-----")

    return pred_label, pred_conf, dict(zip(CLASS_LABELS, probs.tolist()))

def test_folder(folder_path: str, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(extensions):
            predict_image(os.path.join(folder_path, fname))

# ---- EXAMPLES ----
if __name__ == "__main__":
    # Single image test
    img_path = "image_processing20220902-2740614-ioomxn.jpg"
    predict_image(img_path)

    # Folder test (uncomment to run)
    # test_folder("Original Dataset/Healthy Leaf")
