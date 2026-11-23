
# AI Fraud Detection System
# Step 5: Feature Engineering

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Cleaned Dataset
df = pd.read_csv("cleaned_fraudTest.csv.zip")

print("Feature Engineering Started!")
print("--------------------------------")

print("\nOriginal Dataset Shape:", df.shape)

# 2. Extract Date and Time Features
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"],
    errors="coerce"
)

df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
df["transaction_day"] = df["trans_date_trans_time"].dt.day
df["transaction_month"] = df["trans_date_trans_time"].dt.month

# 3. Calculate Customer Age
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

df["customer_age"] = (
    (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
)

# 4. Remove Unnecessary Columns
columns_to_drop = [
    "Unnamed: 0",
    "cc_num",
    "trans_num",
    "first",
    "last",
    "street",
    "city",
    "zip",
    "dob",
    "trans_date_trans_time",
    "unix_time"
]

df.drop(
    columns=columns_to_drop,
    inplace=True,
    errors="ignore"
)

print("\nUnnecessary Columns Removed!")

# 5. Handle Missing Values
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# 6. Label Encoding
categorical_columns = df.select_dtypes(include=["object"]).columns

label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    label_encoders[column] = encoder

print("\nLabel Encoding Completed!")

# 7. Correlation Analysis
correlation = df.corr(numeric_only=True)

fraud_correlation = (
    correlation["is_fraud"]
    .drop("is_fraud")
    .abs()
    .sort_values(ascending=False)
)

print("\nTop Features Related to Fraud:")
print(fraud_correlation.head(10))

# 8. Feature Selection
selected_features = fraud_correlation.head(10).index.tolist()

print("\nSelected Features:")
print(selected_features)

# 9. Feature Scaling
features = df.drop(columns=["is_fraud"])
target = df["is_fraud"]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

df_scaled = pd.DataFrame(
    scaled_features,
    columns=features.columns
)

df_scaled["is_fraud"] = target.values

print("\nFeature Scaling Completed!")

# 10. Final Dataset Information
print("\nFinal Dataset Shape:", df_scaled.shape)

print("\nFinal Dataset Preview:")
print(df_scaled.head())

# 11. Save Feature Engineered Dataset
df_scaled.to_csv(
    "feature_engineered_fraud.csv",
    index=False
)

print("\nFeature Engineering Completed Successfully!")
print("Feature Engineered Dataset Saved Successfully!")
