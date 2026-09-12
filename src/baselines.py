import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("evaluation/golden_set.csv").fillna("")

df = df[df["intent"] != ""].copy()

X = df["customer_message"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline 1: Majority class
majority_class = y_train.mode()[0]
majority_predictions = [majority_class] * len(y_test)

print("BASELINE 1 - MAJORITY CLASS")
print("Majority intent:", majority_class)
print("Accuracy:", accuracy_score(y_test, majority_predictions))

# Baseline 2: TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

print("\nBASELINE 2 - TF-IDF + LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))