# file: export_wrap_and_convert.py
import os
import tensorflow as tf
import tf2onnx

KERAS_PATH = "outputs_20250906_142440/model_final.h5"
ONNX_OUT   = "outputs_20250906_142440/model_final.onnx"
INPUT_SIZE = (256, 256, 3)  # <-- set to your real training size

# ---- Load base model in inference mode ----
tf.keras.backend.set_learning_phase(0)   # ensure BN/Dropout in inference
base = tf.keras.models.load_model(KERAS_PATH)
base.trainable = False

# ---- Detect if base already contains Rescaling/Normalization ----
def has_layer_type(model, types):
    return any(isinstance(l, types) for l in model.layers)

HAS_RESCALE = has_layer_type(base, (tf.keras.layers.Rescaling,))
HAS_NORM    = has_layer_type(base, (tf.keras.layers.Normalization,))

# If last layer is already a Softmax with from_logits=False, don't add another
def ends_with_softmax(model):
    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Softmax):
        return True
    # Dense(activation='softmax') case:
    try:
        return getattr(last, "activation", None) == tf.keras.activations.softmax
    except Exception:
        return False

inp = tf.keras.Input(shape=INPUT_SIZE, name="input")

x = inp
# === Put your REAL training preprocess here ===
# If base already has Rescaling/Normalization inside, don't add again.
if not HAS_RESCALE and not HAS_NORM:
    # OPTION A: simple 1/255 scaling (common if you had a Rescaling layer while training)
    x = tf.keras.layers.Rescaling(1./255.0, name="rescale_1_div_255")(x)

    # OPTION B (uncomment instead of A if you trained with ImageNet mean/std):
    # x = tf.keras.layers.Rescaling(1./255.0, name="rescale")(x)
    # x = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406],
    #                                   variance=[0.229**2, 0.224**2, 0.225**2],
    #                                   name="imagenet_norm")(x)

y = base(x)

# Ensure probabilities out (not raw logits)
if ends_with_softmax(base):
    out = y
else:
    out = tf.keras.layers.Softmax(name="softmax")(y)

wrapped = tf.keras.Model(inp, out, name="wrapped_for_export")

# ---- Convert to ONNX ----
spec = (tf.TensorSpec((None,) + INPUT_SIZE, tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    wrapped,
    input_signature=spec,
    opset=13,              # 13–15 are safe
)

with open(ONNX_OUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Saved:", ONNX_OUT)
