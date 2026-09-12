import pandas as pd

file = "evaluation/golden_set.csv"
df = pd.read_csv(file, dtype=str).fillna("")

def suggest_intent(text):
    t = text.lower()

    if any(x in t for x in ["delivered", "didn't receive", "not received", "never arrived"]):
        return "delivery_missing"
    if any(x in t for x in ["late", "delayed", "delivery date", "when will", "not arrived"]):
        return "delivery_delay"
    if any(x in t for x in ["damaged", "broken", "wrong item", "defective"]):
        return "delivery_wrong_or_damaged"
    if any(x in t for x in ["refund", "return", "replacement"]):
        return "refund_or_return"
    if any(x in t for x in ["payment", "charged", "charge", "price", "cost"]):
        return "payment_or_price"
    if any(x in t for x in ["password", "login", "sign in", "account"]):
        return "account_or_login"
    if any(x in t for x in ["prime", "membership", "subscription"]):
        return "prime_or_subscription"
    if any(x in t for x in ["cancel order", "cancel my order", "change my order"]):
        return "order_change_or_cancellation"
    if any(x in t for x in ["product", "item", "app", "service"]):
        return "product_or_service_issue"

    return "other"


for i in range(len(df)):

    if df.loc[i, "intent"] and df.loc[i, "should_escalate"]:
        continue

    message = df.loc[i, "customer_message"]
    suggestion = suggest_intent(message)

    print("\n" + "=" * 60)
    print(f"EXAMPLE {i + 1} OF {len(df)}")
    print("=" * 60)
    print("\nCUSTOMER:")
    print(message)
    print("\nSUGGESTED INTENT:", suggestion)

    answer = input("Accept? (Enter = yes, type correct intent = no): ").strip()

    if answer:
        df.loc[i, "intent"] = answer
    else:
        df.loc[i, "intent"] = suggestion

    answer = input("Escalate? (yes/no): ").strip().lower()
    df.loc[i, "should_escalate"] = answer

    df.loc[i, "notes"] = "Human-reviewed"

    df.to_csv(file, index=False)

print("\nDONE! 150 examples labelled.")