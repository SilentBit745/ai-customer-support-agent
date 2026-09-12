import pandas as pd
from agent import run_agent

GOLDEN = "evaluation/golden_set.csv"
OUTPUT = "evaluation/results.csv"

df = pd.read_csv(GOLDEN).fillna("")

results = []

for _, row in df.iterrows():
    prediction = run_agent(row["customer_message"])

    results.append({
        "customer_message": row["customer_message"],
        "true_intent": row["intent"],
        "predicted_intent": prediction["intent"],
        "reply": prediction["reply"],
        "decision": prediction["decision"],
        "reason": prediction["reason"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, index=False)

intent_accuracy = (
    results_df["true_intent"] == results_df["predicted_intent"]
).mean()

print("\nEVALUATION RESULTS")
print("------------------")
print(f"Examples evaluated: {len(results_df)}")
print(f"Intent accuracy: {intent_accuracy:.2%}")
print(f"Results saved to: {OUTPUT}")