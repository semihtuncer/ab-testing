
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from layout_features import extract_layout_features
from data_loader import download_ad_layouts

def load_data(path):
    df = pd.read_csv(path)

    required = {"layout_json", "clicks", "impressions"}
    if not required.issubset(df.columns):
        raise ValueError("Missing dataset columns")

    df = df.copy()

    for col in ["clicks", "impressions"]:
        if (df[col].isna().any() or not np.all(df[col] == df[col].astype(int))):
            raise ValueError(f"{col} must contain integers")

    df = df[(df["impressions"] > 0) & (df["clicks"] >= 0) & (df["clicks"] <= df["impressions"])].reset_index(drop=True)

    X = np.stack([extract_layout_features(layout) for layout in df["layout_json"]])

    return df, X

def fit_binomial_model(X, clicks, impressions):
    X_train = np.repeat(X, 2, axis=0)
    y_train = np.tile([1, 0], len(X))

    weights = np.column_stack([clicks, impressions - clicks]).reshape(-1)

    mask = weights > 0
    model = LogisticRegression(C=0.1, max_iter=2000)

    model.fit(X_train[mask], y_train[mask], sample_weight=weights[mask])

    return model

df = download_ad_layouts()

X = np.stack([extract_layout_features(layout) for layout in df["layout_json"]])

train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

model = fit_binomial_model(X[train_idx], train["clicks"].to_numpy(), train["impressions"].to_numpy())

predicted_ctr = model.predict_proba(X[test_idx])[:, 1]

p = np.clip(predicted_ctr, 1e-8, 1 - 1e-8)

clicks = test["clicks"].to_numpy()
impressions = test["impressions"].to_numpy()

nll = -np.sum(clicks * np.log(p) + (impressions - clicks) * np.log(1 - p)) / impressions.sum()

print("Test log loss:", nll)

baseline_ctr = train["clicks"].sum() / train["impressions"].sum()
baseline_p = np.clip(baseline_ctr, 1e-8, 1 - 1e-8)

baseline_nll = -np.sum(clicks * np.log(baseline_p) + (impressions - clicks) * np.log(1 - baseline_p)) / impressions.sum()

print("Baseline log loss:", baseline_nll)
joblib.dump(model, "layout_model.joblib")