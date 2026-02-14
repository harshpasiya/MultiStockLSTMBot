import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from build_lstm_model import build_lstm_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/sequences.npz"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

EPOCHS = 150
BATCH_SIZE = 64


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found.")

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

print(f"Train shape: {X_train.shape}")
print(f"Val shape:   {X_val.shape}")

# Sanity checks
if np.isnan(X_train).any() or np.isnan(y_train).any():
    raise ValueError("NaNs found in training data.")

if np.isnan(X_val).any() or np.isnan(y_val).any():
    raise ValueError("NaNs found in validation data.")

print("\nTarget distribution:")
print(f"Mean y_train: {np.mean(y_train):.6f}")
print(f"Std  y_train: {np.std(y_train):.6f}")


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding model...")

# Dynamically adjust feature size
n_features = X_train.shape[2]

model = build_lstm_model(n_features=n_features)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

checkpoint_path = os.path.join(
    MODELS_DIR,
    f"zodic_omega_best_{run_id}.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,        # Important for pooled multi-stock data
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODELS_DIR,
    f"zodic_omega_final_{run_id}.keras"
)

model.save(final_model_path)

print(f"\nFinal model saved → {final_model_path}")
print(f"Best model saved  → {checkpoint_path}")


# ============================================================
# SAVE TRAINING CURVE
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
plt.legend()
plt.grid(alpha=0.3)

plot_path = os.path.join(
    PLOTS_DIR,
    f"training_history_{run_id}.png"
)

plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()

print(f"Training plot saved → {plot_path}")

print("\nTraining complete.")
