import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "addition_dataset.csv"
)


df = pd.read_csv(file_path)

print("CSV file loaded successfully!")
print("File:", file_path)
print("Total rows:", len(df))


print("\nFirst 5 rows:")
print(df.head())

print("\nCSV Columns:")
print(df.columns)


X = df[["A", "B", "C"]]

y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)



print("\n================================")
print("MODEL RESULTS")
print("================================")

print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


A = 100
B = 200
C = 300

new_data = pd.DataFrame(
    [[A, B, C]],
    columns=["A", "B", "C"]
)

prediction = model.predict(new_data)


print("\n================================")
print("NEW INPUT TEST")
print("================================")

print("A:", A)
print("B:", B)
print("C:", C)

if prediction[0] == 1:
    print("Prediction: Correct Sum")
else:
    print("Prediction: Incorrect Sum")