import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "Genre", "Plot"]
)

print("\nBefore CLEANING DATA(OGDATA):\n")
print(df.head())


stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["Clean_Plot"] = df["Plot"].apply(clean_text)

print("\nCLEANED TEXT:\n")
print(df[["Plot", "Clean_Plot"]].head())

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X = tfidf.fit_transform(df["Clean_Plot"])
y = df["Genre"]

print("\nTF-IDF Shape:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training Naive Bayes Model

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

print("\nNaive Bayes Accuracy:")
print(nb_accuracy)

# Training Logistic Regression Model

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

# Training SVM Model...

svm_model = LinearSVC(C=2)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:")
print(svm_accuracy)


print("\nComparing Models")
print("Naive Bayes Accuracy:", nb_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("SVM:", svm_accuracy)


print("MOVIE GENRE PREDICTION SYSTEM")

while True:
    user_input = input("\nEnter movie plot (or type quit):\n")
    if user_input.lower() == "quit":
        print("\nExiting Program...")
        break
    clean_input = clean_text(user_input)
    input_vector = tfidf.transform([clean_input])
    prediction = lr_model.predict(input_vector)

    print("\nPredicted Genre:")
    print(prediction[0])