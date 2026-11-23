
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Prediction Data
df = pd.read_csv("fraud_predictions.csv")

# Actual Values
y_test = df["Actual_Fraud"]

# All Machine Learning Models
model_columns = [
    "Logistic_Regression",
    "Decision_Tree",
    "Random_Forest",
    "XGBoost"
]

results = []

# Evaluate Each Model
for model in model_columns:

    y_pred = df[model]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": model,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print("\nModel:", model)
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Fraud"],
        yticklabels=["Safe", "Fraud"]
    )

    plt.title(model + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Compare Models
results_df = pd.DataFrame(results)

print("\nFinal Model Evaluation:")
print(results_df)

# Save Results
results_df.to_csv(
    "step8_evaluation_results.csv",
    index=False
)

print("\nStep 8 Evaluation Completed Successfully!")
