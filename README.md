# Transaction-Detection
This project leverages the power of Artificial Intelligence to detect fraudulent and legitimate financial transactions with high accuracy. By analyzing user behavior and transaction patterns, the model helps identify suspicious activities that deviate from normal behavior in real-time.

How It Works
-------------------
Data Collection: The system processes structured transaction data, including features such as amount, timestamp, user demographics, merchant info, and historical patterns.

Feature Engineering: Important numerical and categorical features are scaled and encoded to reflect transactional behavior, including:

Transaction amount

Frequency of transactions

Merchant risk score

Average transaction value over time

Modeling with AI: A pre-trained deep learning model (PyTorch-based) is used to analyze the transaction features and classify each transaction as:

Fraudulent

Legitimate

Real-Time Classification: Transactions are evaluated as soon as they're uploaded, and the predictions are saved into a MySQL database for audit and monitoring.

Visualization: The app includes a dynamic dashboard built with Streamlit, allowing users to:

View historical transactions

Monitor fraud detection stats

Analyze trends over time via interactive graphs

Key Features
-------------------------------------
Secure user authentication (Sign Up / Sign In)

Upload .txt or .json transaction files

Instant fraud prediction

Interactive visual reports

MySQL database integration for persistent storage
