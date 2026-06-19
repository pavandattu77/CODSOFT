import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())

print(df.shape)
df = df.iloc[:, :2]

df.columns = ["label", "message"]

print(df.head())
df["label"] = df["label"].map({"ham": 0,"spam": 1})


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_message"] = df["message"].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))

X = tfidf.fit_transform(df["clean_message"])

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

svm = LinearSVC()
svm.fit(X_train, y_train)    
svm_pred = svm.predict(X_test)

while True:
    sms = input("\nEnter SMS (or quit): ")
    if sms.lower() == "quit":
        break
    sms_clean = clean_text(sms)
    sms_vector = tfidf.transform([sms_clean])
    prediction = svm.predict(sms_vector)
    if prediction[0] == 1:
        print("SPAM")
    else:
        print("HAM")

print("\nCompare Models:")
print("Naive Bayes:", accuracy_score(y_test, nb_pred))
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("SVM:", accuracy_score(y_test, svm_pred))
print("\nNaive Bayes Report")
print(classification_report(y_test, nb_pred))
print("\nLogistic Regression Report")
print(classification_report(y_test, lr_pred))
print("\nSVM Report")
print(classification_report(y_test, svm_pred))