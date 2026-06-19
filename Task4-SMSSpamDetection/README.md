SMS Spam Detection

Project Overview

This project classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Machine Learning.

Dataset

The dataset contains SMS messages labeled as:

- Spam
- Ham

Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn

Machine Learning Models

- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

NLP Techniques Used

- Text Cleaning
- Stopword Removal
- TF-IDF Vectorization

Workflow

1. Load Dataset
2. Text Cleaning
3. TF-IDF Feature Extraction
4. Train-Test Split
5. Model Training
6. Model Comparison
7. Dynamic SMS Prediction

Results

- Naive Bayes Accuracy: 97.22%
- Logistic Regression Accuracy: 95.25%
- SVM Accuracy: 97.85%

Best Model

SVM achieved the highest accuracy and provided the best spam detection performance.

Example

Input:
Congratulations! You have won ₹50,000. Click here to claim your prize.

Output:
SPAM

Input:
Hey bro, are you coming to college today?

Output:
HAM

How to Run

Install dependencies:

pip install -r requirements.txt

Run:

python app.py

Features

- Dynamic SMS Prediction
- Model Comparison
- NLP-Based Text Classification