# AI Customer Support Agent

An NLP based customer support agent that understands customer messages, finds similar historical conversations, and suggests a relevant support response.

## Overview

This project uses real customer support conversations from Twitter to build a lightweight support assistant.

The system can:

- Identify the type of customer problem.
- Find similar past support conversations.
- Suggest a response based on historical support replies.
- Decide whether to handle the issue automatically or send it to a human.

## Brand

The project focuses on Amazon customer support conversations from `@AmazonHelp`.

## Dataset

The project uses the Customer Support on Twitter dataset.

Dataset:
`thoughtvector/customer-support-on-twitter`

Only Amazon conversations are used.

For faster processing, the retrieval system uses a 20,000 row sample of the Amazon conversation data.

## Intent Categories

The following support categories are used:

- Delivery delay
- Missing delivery
- Wrong or damaged item
- Refund or return
- Payment or price
- Account or login
- Prime or subscription
- Product or service issue
- Order change or cancellation
- Other

## How It Works

```text
Customer message
       ↓
Intent classification
       ↓
Find similar conversations
       ↓
Retrieve historical support replies
       ↓
Generate a support response
       ↓
Auto-handle or escalate
