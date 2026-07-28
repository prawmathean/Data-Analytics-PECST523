import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

n = 100
a = np.round(np.random.uniform(1, 100, n), 2)
b = np.round(np.random.uniform(1, 100, n), 2)
c = np.round(a + b, 2)

df = pd.DataFrame({"a": a, "b": b, "c": c})
df.to_csv("dataset.csv", index=False)
print("Sample of dataset:")
print(df.head(), "\n")

X = df[["a", "b"]]
y = df["c"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model coefficients (a, b):", model.coef_)
print("Model intercept:", model.intercept_)
print("R^2 score:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 4))

def predict_sum(a_val, b_val):
    result = model.predict(pd.DataFrame({"a": [a_val], "b": [b_val]}))[0]
    return round(result, 2)

a_input = 23
b_input = 7
print(f"\nPredicted c for a={a_input}, b={b_input}: {predict_sum(a_input, b_input)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred, color="royalblue", label="Test points")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
axes[0].set_xlabel("Actual c")
axes[0].set_ylabel("Predicted c")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

axes[1].scatter(df["a"] + df["b"], df["c"], color="orange", s=20, label="Data (c)")
axes[1].plot(
    [df["a"].add(df["b"]).min(), df["a"].add(df["b"]).max()],
    [df["a"].add(df["b"]).min(), df["a"].add(df["b"]).max()],
    color="green", linestyle="--", label="a+b line"
)
axes[1].set_xlabel("a + b")
axes[1].set_ylabel("c")
axes[1].set_title("Dataset: c vs sum")
axes[1].legend()

plt.tight_layout()
plt.savefig("regression_plots.png", dpi=150)
plt.show()
