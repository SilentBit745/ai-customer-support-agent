import pandas as pd

INPUT_FILE = r"archive\twcs\twcs.csv"
OUTPUT_FILE = r"data\amazon_pairs.csv"

print("Step 1: Finding AmazonHelp replies...")

amazon_parent_ids = set()

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=200000,
):
    amazon = chunk[
        chunk["author_id"].astype(str).eq("AmazonHelp")
    ]

    parent_ids = (
        amazon["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
        .astype(str)
    )

    amazon_parent_ids.update(parent_ids)

print("Customer messages linked to Amazon replies:", len(amazon_parent_ids))

print("Step 2: Finding those customer messages...")

pairs = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=200000,
):
    customers = chunk[
        chunk["tweet_id"].astype(str).isin(amazon_parent_ids)
    ]

    if len(customers) > 0:
        pairs.append(customers)

customers = pd.concat(pairs, ignore_index=True)

print("Customer messages found:", len(customers))

print("Step 3: Loading Amazon replies...")

amazon_replies = []

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=200000,
):
    amazon = chunk[
        chunk["author_id"].astype(str).eq("AmazonHelp")
        & chunk["in_response_to_tweet_id"].notna()
    ]

    if len(amazon) > 0:
        amazon_replies.append(amazon)

amazon = pd.concat(amazon_replies, ignore_index=True)

amazon["customer_id"] = (
    amazon["in_response_to_tweet_id"]
    .astype(int)
    .astype(str)
)

customers["tweet_id"] = customers["tweet_id"].astype(str)

result = amazon.merge(
    customers[["tweet_id", "text"]],
    left_on="customer_id",
    right_on="tweet_id",
    suffixes=("_amazon", "_customer"),
)

result = result[
    ["customer_id", "text_customer", "tweet_id_amazon", "text_amazon"]
]

result.columns = [
    "customer_id",
    "customer_message",
    "amazon_reply_id",
    "amazon_reply",
]

result = result.drop_duplicates()

result.to_csv(OUTPUT_FILE, index=False)

print("\nDONE!")
print("Conversation pairs:", len(result))
print("Saved to:", OUTPUT_FILE)