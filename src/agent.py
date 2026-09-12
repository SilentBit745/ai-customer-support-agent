import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = "data/amazon_pairs.csv"

df = pd.read_csv(DATA).fillna("")
df = df.sample(min(20000, len(df)), random_state=42).reset_index(drop=True)

intents = {
    "delivery_delay": ["late", "delayed", "delivery", "arrive", "shipping"],
    "delivery_missing": ["missing", "not received", "didn't receive", "delivered"],
    "delivery_wrong_or_damaged": ["wrong", "damaged", "broken", "defective"],
    "refund_or_return": ["refund", "return", "replacement"],
    "payment_or_price": ["payment", "price", "charged", "charge", "cost"],
    "account_or_login": ["password", "login", "account", "sign in"],
    "prime_or_subscription": ["prime", "membership", "subscription"],
    "order_change_or_cancellation": ["cancel order", "change order"],
    "product_or_service_issue": ["product", "item", "app", "service"],
    "other": []
}

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000
)

matrix = vectorizer.fit_transform(df["customer_message"])


def classify(message):
    text = message.lower()

    scores = {}

    for intent, words in intents.items():
        scores[intent] = sum(word in text for word in words)

    return max(scores, key=scores.get)


def retrieve(message, k=3):
    query = vectorizer.transform([message])
    scores = cosine_similarity(query, matrix)[0]
    indexes = scores.argsort()[-k:][::-1]

    return df.iloc[indexes]


def run_agent(message):

    intent = classify(message)

    examples = retrieve(message)

    best_reply = examples.iloc[0]["amazon_reply"]

    sensitive_words = [
        "fraud", "scam", "hacked", "stolen",
        "security", "unauthorized", "angry"
    ]

    escalate = any(word in message.lower() for word in sensitive_words)

    if escalate:
        decision = "escalate"
        reason = "Sensitive or potentially high-risk issue."
    else:
        decision = "auto_handle"
        reason = "Similar historical support cases were found."

    return {
        "intent": intent,
        "reply": best_reply,
        "decision": decision,
        "reason": reason,
        "historical_examples": examples[
            ["customer_message", "amazon_reply"]
        ].to_dict("records")
    }


if __name__ == "__main__":

    message = input("Customer message: ")

    result = run_agent(message)

    print("\nRESULT")
    print("Intent:", result["intent"])
    print("Reply:", result["reply"])
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])

    print("\nHistorical evidence:")
    for example in result["historical_examples"]:
        print("\nCustomer:", example["customer_message"])
        print("Amazon:", example["amazon_reply"])