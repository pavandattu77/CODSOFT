Customer Churn Prediction

Project Overview

This project predicts whether a customer is likely to leave a bank based on customer information and account details.

Dataset

The dataset contains customer demographics, account information, and churn status ("Exited").

Technologies Used

- Python
- Pandas
- Scikit-learn

Machine Learning Models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Features Used

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member Status
- Estimated Salary

Data Preprocessing

- Removed unnecessary columns:
  - RowNumber
  - CustomerId
  - Surname
- Label Encoding applied to:
  - Geography
  - Gender

Workflow

1. Load Dataset
2. Data Cleaning
3. Label Encoding
4. Train-Test Split
5. Model Training
6. Model Comparison

Results

- Logistic Regression Accuracy: 80.65%
- Random Forest Accuracy: 86.40%
- SVM Accuracy: 79.65%

Best Model

Random Forest achieved the highest accuracy.

How to Run

Install dependencies:

pip install -r requirements.txt

Run:

python app.py

Output

Predicts whether a customer is likely to:

- Stay
- Exit the bank