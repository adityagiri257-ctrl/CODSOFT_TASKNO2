# Credit Card Fraud Detection using Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
# Dataset should contain a column named 'Class'
# Class = 0 (Legitimate), 1 (Fraud)
data = pd.read_csv("creditcard.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict a single transaction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction: Fraudulent Transaction")
else:
    print("\nPrediction: Legitimate Transaction")
