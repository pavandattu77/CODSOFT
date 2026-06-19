import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

df = pd.read_csv("Churn_Modelling.csv")

print(df.head())
print(df.shape)
print(df["Exited"].value_counts())

df = df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname"
    ]
 )

le_geo = LabelEncoder()
le_gender = LabelEncoder()

df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"] = le_gender.fit_transform(df["Gender"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:",accuracy_score(y_test, lr_pred))

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",accuracy_score(y_test, rf_pred))

svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("SVM Accuracy:",accuracy_score(y_test, svm_pred))
print("\nComparing Models:")
print("Logistic Regression:",accuracy_score(y_test, lr_pred))
print("Random Forest:",accuracy_score(y_test, rf_pred))
print("SVM:",accuracy_score(y_test, svm_pred))
print("\nCUSTOMER CHURN PREDICTION")
print("=========================")

credit_score = int(input("Credit Score: "))

print("\nGeography:")
print("0 = France")
print("1 = Germany")
print("2 = Spain")
geography = int(input("Enter Geography: "))

print("\nGender:")
print("0 = Female")
print("1 = Male")
gender = int(input("Enter Gender: "))

age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = float(input("Balance: "))
num_products = int(input("Number of Products: "))
has_card = int(input("Has Credit Card? (0=No, 1=Yes): "))
active_member = int(input("Is Active Member? (0=No, 1=Yes): "))
salary = float(input("Estimated Salary: "))

customer = [[credit_score, geography, gender, age, tenure,balance,num_products,has_card,active_member,salary]]

prediction = rf.predict(customer)

print("\nRESULT")
print("======")

if prediction[0] == 1:
    print("⚠ Customer is likely to LEAVE the bank")
else:
    print("✅ Customer is likely to STAY with the bank")