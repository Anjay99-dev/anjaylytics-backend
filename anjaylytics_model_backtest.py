# anjaylytics_model_backtest.py
# -------------------------------------------------------------
# Trains a simple classifier on yfinance daily features
# Saves:
#   models/anjaylytics_model.pkl
#   models/backtest_metrics.json  (includes calibration bins)
# -------------------------------------------------------------

import os, json, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from joblib import dump

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

WATCH = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","SPY","QQQ"]

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please: pip install yfinance pandas scikit-learn joblib numpy") from e

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret1"] = x["Close"].pct_change()
    x["ret5"] = x["Close"].pct_change(5)
    x["ret20"] = x["Close"].pct_change(20)
    x["vol20"] = x["ret1"].rolling(20).std()
    x["atr10"] = (x["High"] - x["Low"]).rolling(10).mean() / x["Close"].shift(1)
    x["gap1"] = (x["Open"] - x["Close"].shift(1)) / x["Close"].shift(1)
    x["mom50"] = x["Close"].pct_change(50)
    return x

def label_target(df: pd.DataFrame, horizon=5, up=0.02, down=0.02):
    # Binary target: +1 if max future return over next h days >= up
    # (you can refine to include down side or more nuanced objectives)
    fwd = df["Close"].pct_change().shift(-1).rolling(horizon).sum()
    y = (fwd >= up).astype(int)
    return y

def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty: return pd.DataFrame()
    return df

def build_dataset():
    frames = []
    for t in WATCH:
        df = load_data(t)
        if df.empty: continue
        feats = make_features(df)
        y = label_target(df)
        feats["y"] = y
        feats["ticker"] = t
        frames.append(feats)
    data = pd.concat(frames, axis=0, ignore_index=False)
    data = data.dropna()
    feats = ["ret1","ret5","ret20","vol20","atr10","gap1","mom50"]
    X = data[feats].values
    y = data["y"].values.astype(int)
    return X, y, feats, data.index

def main():
    X, y, feats, idx = build_dataset()
    n = len(y)
    if n < 200:
        raise SystemExit("Not enough data to train; expand WATCH or period.")

    # Simple split (time-ordered)
    split = int(n * 0.75)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    # Predict proba on test
    p_te = model.predict_proba(X_te)[:,1]
    brier = float(brier_score_loss(y_te, p_te))

    # Calibration bins
    prob_true, prob_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    bins = []
    # derive counts per bin approximately (using histogram on predictions)
    hist, edges = np.histogram(p_te, bins=10)
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        bins.append({"p_avg": float(pp), "y_rate": float(pt), "n": int(hist[i] if i < len(hist) else 0)})

    # Save artifacts
    dump(model, os.path.join(OUT_DIR, "anjaylytics_model.pkl"))
    metrics = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "brier_overall": brier,
        "calibration": bins,
        "by_month": [],  # optional: add grouped PnL metrics later
        "notes": "LogisticRegression on daily features; 75/25 time split; horizon=5 up=2%."
    }
    with open(os.path.join(OUT_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", os.path.join(OUT_DIR, "anjaylytics_model.pkl"))
    print("Saved:", os.path.join(OUT_DIR, "backtest_metrics.json"))
    print("Brier (test):", brier)

if __name__ == "__main__":
    main()
