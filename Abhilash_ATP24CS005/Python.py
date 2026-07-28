import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

A = np.random.randint(1, 100, 100)
B = np.random.randint(1, 100, 100)

C = A + B

error_rows = np.random.choice(100, 10, replace=False)

for i in error_rows:
    C[i] += np.random.randint(-10, 10)

data = pd.DataFrame({
    "A": A,
    "B": B,
    "C": C
})

data.to_csv("sum_dataset.csv", index=False)

print("Dataset created with 10% incorrect rows")

data = pd.read_csv("sum_dataset.csv")

X = data[["A", "B"]]
Y = data["C"]

model = LinearRegression()
model.fit(X, Y)

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

new_data = pd.DataFrame({
    "A": [a],
    "B": [b]
})

result = model.predict(new_data)

print("\nPredicted Sum =", round(result[0], 2))
