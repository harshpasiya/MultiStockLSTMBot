import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from build_lstm_model import build_lstm_model


# ============================================================
# REPRODUCIBILITY (IMPORTANT)
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

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print(f"Train shape: {X_train.shape}")
print(f"Val shape:   {X_val.shape}")
print(f"Test shape:  {X_test.shape}")


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding model...")
model = build_lstm_model()
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
# EVALUATE ON TEST SET
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Test MSE : {test_loss:.6f}")
print(f"Test MAE : {test_mae:.6f}")


# ============================================================
# SAVE TRAINING CURVE
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
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
