import pandas as pd
from agent import run_agent

INPUT = "evaluation/golden_set.csv"
OUTPUT = "evaluation/results.csv"

data = pd.read_csv(INPUT).fillna("")

results = []

for _, row in data.iterrows():
    result = run_agent(row["customer_message"])

    results.append({
        "customer_message": row["customer_message"],
        "true_intent": row["intent"],
        "predicted_intent": result["intent"],
        "reply": result["reply"],
        "decision": result["decision"],
        "reason": result["reason"]
    })

results = pd.DataFrame(results)
results.to_csv(OUTPUT, index=False)

accuracy = (
    results["true_intent"] == results["predicted_intent"]
).mean()

print("\nEVALUATION RESULTS")
print("------------------")
print("Examples evaluated:", len(results))
print(f"Intent accuracy: {accuracy:.2%}")
print("Results saved to:", OUTPUT)