# SMS Spam Detection using TF-IDF and Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
# Dataset should contain two columns: label and message
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['label', 'message']]

# Convert labels to numeric values
# ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with a custom message
new_message = ["Congratulations! You have won a free mobile. Claim your prize now."]

new_message = vectorizer.transform(new_message)

prediction = model.predict(new_message)

if prediction[0] == 1:
    print("\nPrediction: Spam")
else:
    print("\nPrediction: Legitimate (Ham)")
