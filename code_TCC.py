# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Dataset1.csv")

# Display first rows
print(df.head())

# -----------------------------
# DATA PREPROCESSING
# -----------------------------

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -----------------------------
# SPLIT DATA
# -----------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# VISUALIZATION
# -----------------------------

sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()