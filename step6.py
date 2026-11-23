
# AI Fraud Detection System
# Step 6: Machine Learning Model Building

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Machine Learning Model Building Started!")
print("------------------------------------------")

# 1. Load Cleaned Dataset
df = pd.read_csv("cleaned_fraudTest.csv.zip")

# 2. Extract Date and Time Features
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"],
    errors="coerce"
)

df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
df["transaction_day"] = df["trans_date_trans_time"].dt.day
df["transaction_month"] = df["trans_date_trans_time"].dt.month

# 3. Remove Unnecessary Columns
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

# 4. Separate Features and Target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 5. Identify Numerical and Categorical Columns
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# 7. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Define Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# 9. Train and Evaluate Models
results = []

for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(model_name, "Training Completed!")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

# 10. Compare Model Performance
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

# 11. Save Results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel Building Completed Successfully!")
print("Model Comparison Saved Successfully!")
