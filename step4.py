
# AI Fraud Detection System
# Step 4: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Cleaned Dataset
df = pd.read_csv("cleaned_fraudTest.csv.zip")

# Convert Date Column
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"],
    errors="coerce"
)

print("EDA Started Successfully!")
print("--------------------------------")

# 1. Basic Dataset Information
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 2. Fraud vs Non-Fraud Transactions
print("\nFraud Transaction Count:")
print(df["is_fraud"].value_counts())

fraud_count = df["is_fraud"].sum()
total_transactions = len(df)
safe_count = total_transactions - fraud_count

fraud_percentage = (fraud_count / total_transactions) * 100

print("\nTotal Transactions:", total_transactions)
print("Fraud Transactions:", fraud_count)
print("Safe Transactions:", safe_count)
print("Fraud Percentage:", round(fraud_percentage, 2), "%")

# 3. Transaction Amount Analysis
print("\nAverage Transaction Amount:")
print(df.groupby("is_fraud")["amt"].mean())

# 4. State-wise Fraud Analysis
print("\nTop 10 States with Fraud Transactions:")
state_fraud = df[df["is_fraud"] == 1]["state"].value_counts().head(10)
print(state_fraud)

# 5. Merchant-wise Fraud Analysis
print("\nTop 10 Merchants with Fraud Transactions:")
merchant_fraud = (
    df[df["is_fraud"] == 1]["merchant"].value_counts().head(10)
)
print(merchant_fraud)

# 6. Time-based Fraud Analysis
df["transaction_hour"] = df["trans_date_trans_time"].dt.hour

print("\nFraud Transactions by Hour:")
hour_fraud = df[df["is_fraud"] == 1]["transaction_hour"].value_counts().sort_index()
print(hour_fraud)

# -------------------------------
# DATA VISUALIZATION
# -------------------------------

sns.set_style("whitegrid")

# Graph 1: Fraud vs Non-Fraud Bar Chart
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="is_fraud")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Transaction Type (0 = Safe, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# Graph 2: Fraud Percentage Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    [safe_count, fraud_count],
    labels=["Safe", "Fraud"],
    autopct="%1.2f%%",
    startangle=90
)
plt.title("Fraud Transaction Percentage")
plt.show()

# Graph 3: Transaction Amount Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="amt", bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Graph 4: Transaction Amount Box Plot
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="is_fraud", y="amt")
plt.title("Transaction Amount by Fraud Status")
plt.xlabel("0 = Safe, 1 = Fraud")
plt.ylabel("Transaction Amount")
plt.ylim(0, df["amt"].quantile(0.99))
plt.show()

# Graph 5: State-wise Fraud Count
plt.figure(figsize=(10, 5))
state_fraud.sort_values().plot(kind="bar")
plt.title("Top 10 States with Fraud Transactions")
plt.xlabel("State")
plt.ylabel("Fraud Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graph 6: Merchant-wise Fraud Count
plt.figure(figsize=(10, 5))
merchant_fraud.sort_values().plot(kind="bar")
plt.title("Top 10 Merchants with Fraud Transactions")
plt.xlabel("Merchant")
plt.ylabel("Fraud Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graph 7: Hour-wise Fraud Count
plt.figure(figsize=(10, 5))
hour_fraud.sort_index().plot(kind="bar")
plt.title("Fraud Transactions by Hour")
plt.xlabel("Transaction Hour")
plt.ylabel("Fraud Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Graph 8: Correlation Heatmap
numeric_columns = [
    "amt", "lat", "long", "city_pop",
    "unix_time", "merch_lat", "merch_long", "is_fraud"
]

plt.figure(figsize=(10, 7))
correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Transaction Data Correlation Heatmap")
plt.tight_layout()
plt.show()

# Graph 9: Fraud Count Plot
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="is_fraud", hue="is_fraud", legend=False)
plt.title("Fraud Detection Count Plot")
plt.xlabel("Fraud Status")
plt.ylabel("Transaction Count")
plt.show()

print("\nEDA Completed Successfully!")
