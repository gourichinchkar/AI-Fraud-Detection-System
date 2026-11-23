# Load and explore

import pandas as pd

df = pd.read_csv("fraudTest.csv.zip", index_col=0)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\n", df.head())
print("\n", df.info())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\n", df["is_fraud"].value_counts())
print("\nFraud %:", round(df["is_fraud"].mean() * 100, 3))