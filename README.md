# AI Customer Support Agent

## Hiver SDE Intern Take-Home Assignment

An AI/NLP-based customer support agent built using historical Amazon customer-support conversations from the Customer Support on Twitter dataset.

The system:
1. Classifies a customer message into a support intent.
2. Retrieves similar historical Amazon support conversations.
3. Drafts a reply using historical support responses.
4. Decides whether the issue should be auto-handled or escalated.

## Brand Selected

Amazon Help (@AmazonHelp)

Amazon was selected because it has a large number of customer-support conversations in the dataset.

## Dataset

Customer Support on Twitter dataset:

`thoughtvector/customer-support-on-twitter`

Only Amazon conversations were used.

The full dataset was not processed during inference. A 20,000-row sample of Amazon customer conversations was used for retrieval to keep execution fast.

## Intents

The system uses these intents:

- delivery_delay
- delivery_missing
- delivery_wrong_or_damaged
- refund_or_return
- payment_or_price
- account_or_login
- prime_or_subscription
- product_or_service_issue
- order_change_or_cancellation
- other

## System Pipeline

Customer message
        ↓
Intent classification
        ↓
TF-IDF similarity search
        ↓
Retrieve similar historical Amazon conversations
        ↓
Use historical Amazon response as draft
        ↓
Auto-handle / Escalate decision

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt