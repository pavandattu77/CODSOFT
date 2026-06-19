Credit Card Fraud Detection

Project Overview

This project predicts whether a credit card transaction is fraudulent or legitimate using Machine Learning techniques.

Dataset

The dataset contains over 1.2 million credit card transactions with transaction-related features and a target column named "is_fraud".

Technologies Used

- Python
- Pandas
- Scikit-learn

Machine Learning Models

- Logistic Regression
- Random Forest

Features Used

- Transaction Amount
- Customer Latitude
- Customer Longitude
- City Population
- Merchant Latitude
- Merchant Longitude

Workflow

1. Load Dataset
2. Data Analysis
3. Feature Selection
4. Train-Test Split
5. Model Training
6. Performance Evaluation

Challenge

The dataset is highly imbalanced, with very few fraud transactions compared to legitimate transactions.

Solution

Used "class_weight="balanced"" in Logistic Regression to improve fraud detection performance.

Results

- Logistic Regression: Successfully detected fraudulent transactions after handling class imbalance.
- Random Forest: Used for model comparison and performance improvement.

How to Run

Install dependencies:

pip install -r requirements.txt

Run:

python app.py

Output

The program predicts whether a transaction is:

- Fraudulent
- Legitimate