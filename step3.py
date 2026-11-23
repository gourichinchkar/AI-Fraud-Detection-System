
# AI Fraud Detection System
# Step 3: Data Cleaning

import pandas as pd
import numpy as np

# Load Original Dataset
df = pd.read_csv("fraudTest.csv.zip")

print("Original Dataset Shape:", df.shape)

# 1. Remove Unnecessary Columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nUnnecessary Columns Removed")

# 2. Check Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 3. Remove Duplicate Records
duplicate_count = df.duplicated().sum()

print("\nDuplicate Records Found:", duplicate_count)

df = df.drop_duplicates()

print("Duplicate Records Removed")

# 4. Handle Missing Values

# Handle missing values in Target Column
if "is_fraud" in df.columns:
    df = df.dropna(subset=["is_fraud"])

# Fill missing numerical values with Median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with Mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    if df[column].isnull().any():
        mode_value = df[column].mode()
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])

# 5. Convert Date Column
if "trans_date_trans_time" in df.columns:
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"],
        errors="coerce"
    )

# 6. Final Missing Value Check
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# 7. Final Dataset Shape
print("\nCleaned Dataset Shape:", df.shape)

# 8. Save Cleaned Dataset
df.to_csv("cleaned_fraudTest.csv.zip", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned Dataset Saved Successfully!")
