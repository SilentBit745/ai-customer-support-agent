import pandas as pd

df = pd.read_csv("data/amazon_pairs.csv")

golden = df.sample(200, random_state=42)[
    ["customer_message", "amazon_reply"]
].copy()

golden["intent"] = ""
golden["should_escalate"] = ""
golden["notes"] = ""

golden.to_csv("evaluation/golden_set.csv", index=False)

print("Golden set created: 200 examples")