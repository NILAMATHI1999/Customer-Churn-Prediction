import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset and preprocess (same steps as training)
df = pd.read_csv("customer_churn.csv")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into train and test sets with same random_state as training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = joblib.load('churn_model.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Print first 10 predictions with input features and actual labels
print("\nFirst 10 test samples with predictions:\n")
for i in range(10):
    print(f"Sample {i+1}:")
    print(X_test.iloc[i])
    print(f"Predicted Churn: {y_pred[i]}")
    print(f"Actual Churn: {y_test.iloc[i]}")
    print("-" * 40)

# Save predictions to CSV
output_df = X_test.copy()
output_df['Predicted_Churn'] = y_pred
output_df['Actual_Churn'] = y_test.values

output_df.to_csv("churn_predictions.csv", index=False)
print("\n✅ Predictions saved to churn_predictions.csv")

# --- Step 2: Evaluate the model ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n✅ Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save the plot
plt.show()

print("📊 Confusion matrix plot saved as 'confusion_matrix.png'")
