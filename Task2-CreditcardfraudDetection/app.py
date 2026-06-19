import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fraudTrain.csv")
print(df.head())
print(df.shape)
print(df["is_fraud"].value_counts())
features = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long"
]
X = df[features]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
lr = LogisticRegression(max_iter=1000,class_weight="balanced")
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression:")
print(classification_report(y_test, lr_pred))

rf = RandomForestClassifier(n_estimators=20,random_state=42,n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest:")
print(classification_report(y_test,rf_pred))