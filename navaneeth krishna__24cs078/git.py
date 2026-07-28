import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_csv("dataset.csv")


features = dataset[["a", "b"]]
target = dataset["sum"]


train_features, test_features, train_target, test_target = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=10
)


lr_model = LinearRegression()


lr_model.fit(train_features, train_target)


predictions = lr_model.predict(test_features)


plt.figure(figsize=(7,5))
plt.scatter(test_target, predictions, color="green", marker="o")
plt.plot(
    [test_target.min(), test_target.max()],
    [test_target.min(), test_target.max()],
    color="red",
    linestyle="--",
    linewidth=2
)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.grid(True)
plt.show()


mse = mean_squared_error(test_target, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test_target, predictions)


print("\nPrediction")
first_num = float(input("Enter first number: "))
second_num = float(input("Enter second number: "))


test_data = pd.DataFrame(
    {
        "a": [first_num],
        "b": [second_num]
    }
)


predicted_sum = lr_model.predict(test_data)


print("\nRESULT")
print(f"Predicted Sum : {predicted_sum[0]:.2f}")
print(f"Actual Sum    : {first_num + second_num:.2f}")

print("\nMODEL PERFORMANCE")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")