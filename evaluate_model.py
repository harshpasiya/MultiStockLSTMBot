import numpy as np
import tensorflow as tf
import glob
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/sequences.npz"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test dataset...")

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]
ticker_test = data["ticker_test"]

print(f"Test shape: {X_test.shape}")


# ============================================================
# LOAD LATEST BEST MODEL
# ============================================================

model_files = sorted(glob.glob("models/zodic_omega_best_*.keras"))

if len(model_files) == 0:
    raise ValueError("No trained model found in models/")

MODEL_PATH = model_files[-1]

print(f"\nLoading model → {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating on test set...")

y_pred = model.predict(X_test, verbose=0).flatten()

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss : {test_loss:.6f}")
print(f"Test MAE  : {test_mae:.6f}")


# ============================================================
# BASIC METRICS
# ============================================================

directional_accuracy = np.mean(
    np.sign(y_pred) == np.sign(y_test)
)

ic = np.corrcoef(y_pred, y_test)[0, 1]

rank_ic, _ = spearmanr(y_pred, y_test)

strategy_returns = np.sign(y_pred) * y_test

sharpe_like = (
    np.mean(strategy_returns) /
    (np.std(strategy_returns) + 1e-8)
)


# ============================================================
# PER-STOCK IC ANALYSIS
# ============================================================

df_eval = pd.DataFrame({
    "ticker": ticker_test,
    "y_pred": y_pred,
    "y_test": y_test
})

stock_ics = []

for ticker in df_eval["ticker"].unique():
    sub = df_eval[df_eval["ticker"] == ticker]

    if len(sub) > 30:
        ic_stock = np.corrcoef(sub["y_pred"], sub["y_test"])[0, 1]
        stock_ics.append(ic_stock)

mean_stock_ic = np.nanmean(stock_ics)
std_stock_ic = np.nanstd(stock_ics)


# ============================================================
# LONG–SHORT SIMULATION
# ============================================================

# Top 20% long, bottom 20% short
percentile = 0.2

df_eval["rank"] = df_eval["y_pred"].rank(pct=True)

long_mask = df_eval["rank"] >= (1 - percentile)
short_mask = df_eval["rank"] <= percentile

ls_returns = df_eval.loc[long_mask, "y_test"].mean() - \
             df_eval.loc[short_mask, "y_test"].mean()

# pseudo daily Sharpe approximation
ls_sharpe = (
    ls_returns /
    (df_eval["y_test"].std() + 1e-8)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ TRADING METRICS ================")
print(f"Directional Accuracy : {directional_accuracy:.2%}")
print(f"Information Coef (IC): {ic:.4f}")
print(f"Rank IC              : {rank_ic:.4f}")
print(f"Sharpe-like Metric   : {sharpe_like:.4f}")
print("-------------------------------------------------")
print(f"Mean Stock IC        : {mean_stock_ic:.4f}")
print(f"Stock IC Dispersion  : {std_stock_ic:.4f}")
print("-------------------------------------------------")
print(f"Long–Short Return    : {ls_returns:.4f}")
print(f"Long–Short Sharpe    : {ls_sharpe:.4f}")
print("=================================================")

print("\nMean y_test:", np.mean(y_test))
print("Mean y_pred:", np.mean(y_pred))
