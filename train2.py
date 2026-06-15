# lemon_densenet121_train.py
# Strong baseline for LLDD (Kaggle) using transfer learning + fine-tuning.

import os, pathlib, io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -----------------------------
# Paths & basic config
# -----------------------------
DATA_DIR = "Original Dataset"  # root folder with subfolders per class
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
VAL_SPLIT = 0.1
SEED = 123
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 25   # total epochs ≈ WARMUP_EPOCHS + FINETUNE_EPOCHS
BASE_LR = 4e-4
FINETUNE_LR = 1e-5
OUTPUT_DIR = "outputs_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mixed precision can speed up on modern GPUs (safe to keep off on pure CPU)
# tf.keras.mixed_precision.set_global_policy("mixed_float16")

# -----------------------------
# tf.data pipeline (recommended)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Cache → shuffle → prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# Data augmentation (on-GPU)
# -----------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

# -----------------------------
# Build model (DenseNet121 backbone + custom head)
# -----------------------------
base_model = tf.keras.applications.DenseNet121(
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # freeze for warmup

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="densenet121_lldd")

# Loss with label smoothing helps generalization on small datasets
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BASE_LR),
    loss=loss_fn,
    metrics=["accuracy"]
)

# Convert integer labels to one-hot inside the pipeline (for label smoothing)
def one_hot(ds):
    return ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
train_ds_oh = one_hot(train_ds)
val_ds_oh = one_hot(val_ds)

# -----------------------------
# Callbacks
# -----------------------------
ckpt_path = os.path.join(OUTPUT_DIR, "model_best.h5")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
    )
]

# -----------------------------
# Phase 1: Warm-up (frozen backbone)
# -----------------------------
history_warm = model.fit(
    train_ds_oh,
    validation_data=val_ds_oh,
    epochs=WARMUP_EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# Phase 2: Fine-tuning (unfreeze top layers of DenseNet)
# -----------------------------
# Unfreeze the last dense block for fine-tuning
fine_tune_at = 350  # DenseNet121 has 429 layers; adjust if you like
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= fine_tune_at)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss=loss_fn,
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds_oh,
    validation_data=val_ds_oh,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# Save final model + summary
# -----------------------------
final_path = os.path.join(OUTPUT_DIR, "model_final.h5")
model.save(final_path)
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# Try to save a model architecture diagram (requires pydot + graphviz)
try:
    tf.keras.utils.plot_model(model,
                              to_file=os.path.join(OUTPUT_DIR, "model_arch.png"),
                              show_shapes=True, expand_nested=True, dpi=140)
except Exception as e:
    print("plot_model failed (install graphviz & pydot to enable):", e)

# -----------------------------
# Plot training curves
# -----------------------------
def merge_hist(h1, h2):
    # Join warm + finetune histories for continuous plots
    out = {}
    for k in set(list(h1.history.keys()) + list(h2.history.keys())):
        out[k] = h1.history.get(k, []) + h2.history.get(k, [])
    return out

hist = merge_hist(history_warm, history_ft)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist["accuracy"], label="Train Acc")
plt.plot(hist["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist["loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# -----------------------------
# Confusion matrix on validation set
# -----------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Build a numpy set of validation images/labels
y_true, y_pred = [], []
for batch_x, batch_y in val_ds:
    # raw integer labels from val_ds
    preds = model.predict(batch_x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(batch_y.numpy())

cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print(f"\nAll artifacts saved in: {OUTPUT_DIR}")
print("Best weights:", ckpt_path)
print("Final model :", final_path)
