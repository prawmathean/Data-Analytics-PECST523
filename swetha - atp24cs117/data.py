import pandas as pd
import numpy as np


np.random.seed(25)

num1 = np.random.randint(1, 100, 100)
num2 = np.random.randint(1, 100, 100)

sum_values = num1 + num2

dataset = pd.DataFrame({
    "a": num1,
    "b": num2,
    "c": sum_values
})

print("Dataset:")
print(dataset.head(10))


error_index = np.random.choice(dataset.index, size=10, replace=False)

dataset.loc[error_index, "c"] = dataset.loc[error_index, "c"] + np.random.randint(-5, 6, size=10)

print("Dataset with Errors:")
print(dataset.head(15))

from sklearn.linear_model import LinearRegression


X = dataset[["a", "b"]]
y = dataset["c"]


model = LinearRegression()


model.fit(X, y)


value1 = int(input("Enter value of a: "))
value2 = int(input("Enter value of b: "))

result = model.predict([[value1, value2]])

print("Predicted Sum =", result[0])

import matplotlib.pyplot as plt


predicted = model.predict(X)

plt.figure(figsize=(7,5))

plt.scatter(y, predicted, color="blue", alpha=0.7, label="Predicted Points")

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linewidth=2,
    label="Ideal Line"
)

plt.title("Actual Sum vs Predicted Sum")
plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.legend()
plt.grid(True)

plt.show()