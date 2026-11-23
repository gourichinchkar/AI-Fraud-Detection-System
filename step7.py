
# AI Fraud Detection System
# Step 7: Model Training & Prediction

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Model Training and Prediction Started!")
print("----------------------------------------")

# 1. Load Dataset
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
    "Unnamed: 0", "cc_num", "trans_num",
    "first", "last", "street", "city",
    "zip", "dob", "trans_date_trans_time",
    "unix_time"
]

df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# 4. Separate Features and Target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 6. Identify Column Types
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 7. Data Preprocessing
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

# 8. Define Models
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

# 9. Add XGBoost if Installed
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    print("\nXGBoost Library Available!")

except ImportError:
    print("\nXGBoost Not Installed. Skipping XGBoost.")

# 10. Train Models and Generate Predictions
results = []
prediction_data = pd.DataFrame({
    "Actual_Fraud": y_test.reset_index(drop=True)
})

for model_name, model in models.items():

    print("\nTraining Model:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict Test Transactions
    predictions = pipeline.predict(X_test)

    # Save Predictions
    prediction_data[model_name.replace(" ", "_")] = predictions

    # Calculate Performance
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    # Save Trained Model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(pipeline, model_filename)

    print(model_name, "Training Completed!")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# 11. Model Comparison
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

# 12. Save Model Comparison
results_df.to_csv("step7_model_comparison.csv", index=False)

# 13. Save Prediction Results
prediction_data.to_csv(
    "fraud_predictions.csv",
    index=False
)

print("\nModel Training and Prediction Completed!")
print("Trained Models Saved Successfully!")
print("Prediction Results Saved Successfully!")
