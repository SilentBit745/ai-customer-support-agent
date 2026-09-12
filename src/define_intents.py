import pandas as pd

df = pd.read_csv("data/amazon_pairs.csv")

print("Total conversation pairs:", len(df))

print("\nSample customer messages:\n")

for i, text in enumerate(df["customer_message"].sample(30, random_state=42), 1):
    print(f"{i}. {text}")