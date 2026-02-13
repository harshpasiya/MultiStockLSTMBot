import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# MODEL CONFIGURATION (LOCK FOR PHASE-1)
# ============================================================

LOOKBACK = 30
N_FEATURES = 9

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32

DROPOUT_RATE = 0.2
DENSE_UNITS = 32

LEARNING_RATE = 0.001
GRAD_CLIP_NORM = 1.0


# ============================================================
# MODEL BUILDER
# ============================================================

def build_lstm_model():
    """
    Stable multi-stock shared LSTM model.
    Designed for 5-day forward return regression.
    Production baseline architecture.
    """

    inputs = Input(shape=(LOOKBACK, N_FEATURES), name="price_sequence")

    # First LSTM layer
    x = LSTM(
        LSTM_UNITS_1,
        return_sequences=True,
        name="lstm_layer_1"
    )(inputs)

    x = LayerNormalization(name="layer_norm_1")(x)
    x = Dropout(DROPOUT_RATE, name="dropout_1")(x)

    # Second LSTM layer
    x = LSTM(
        LSTM_UNITS_2,
        return_sequences=False,
        name="lstm_layer_2"
    )(x)

    x = LayerNormalization(name="layer_norm_2")(x)
    x = Dropout(DROPOUT_RATE, name="dropout_2")(x)

    # Dense head
    x = Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x = Dropout(0.2, name="dropout_3")(x)

    outputs = Dense(1, activation="linear", name="forward_return")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ZODIC_OMEGA_LSTM_v1")

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=GRAD_CLIP_NORM
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# EXPORT TEMPLATE MODEL
# ============================================================

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    model = build_lstm_model()
    model.summary()

    model.save("models/zodic_omega_lstm_v1_template.keras")

    print("\nModel template saved → models/zodic_omega_lstm_v1_template.keras")
