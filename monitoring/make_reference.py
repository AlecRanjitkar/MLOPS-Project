import pandas as pd

df = pd.read_csv("prediction_log.csv")

# keep only feature columns
ref = df[["mean", "std", "min", "max"]].head(200)  # pick 200 as baseline

ref.to_csv("reference_features.csv", index=False)
print("✅ Wrote monitoring/reference_features.csv")
