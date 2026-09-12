import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("evaluation/golden_set.csv").fillna("")

data = data[data["intent"].str.strip() != ""]

X = data["customer_message"]
y = data["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline 1: always predict the most common intent
common_intent = y_train.value_counts().idxmax()
majority_predictions = [common_intent] * len(y_test)

print("BASELINE 1 - MAJORITY CLASS")
print("Majority intent:", common_intent)
print("Accuracy:", accuracy_score(y_test, majority_predictions))

# Baseline 2: TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("\nBASELINE 2 - TF-IDF + LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))