import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
data = pd.read_csv("dataset.csv")

# Features and Target
X = data[['a', 'b']]
y = data['sum']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# User Input
print("\n===== TEST THE MODEL =====")

a = float(input("Enter 1st Number : "))
b = float(input("Enter 2nd Number : "))

# Predict
prediction = model.predict([[a, b]])

# Output
print("\n===== OUTPUT =====")
print(f"Predicted Sum : {prediction[0]:.2f}")
print(f"Actual Sum    : {a + b:.2f}")
