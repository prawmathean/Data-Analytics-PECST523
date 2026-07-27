import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)

n_samples = 100

a = np.random.randint(1, 26, size=n_samples)
b = np.random.randint(1, 26, size=n_samples)

true_sum = a + b

noise = np.random.uniform(-0.1, 0.1, size=n_samples)
c = true_sum * (1 + noise)

df = pd.DataFrame({
    "a": a,
    "b": b,
    "c": c
})

df.to_csv("sum_dataset.csv", index=False)
print("Saved dataset to sum_dataset.csv")

X = df[["a", "b"]]   
y = df["c"]          

model = LinearRegression()
model.fit(X, y)

print("Model trained.")
print("Coefficients (for a, b):", model.coef_)
print("Intercept:", model.intercept_)

def predict_sum(a_value, b_value):
    """
    Predict c for given a, b using the trained model.
    """
    X_new = np.array([[a_value, b_value]])
    c_pred = model.predict(X_new)[0]
    return c_pred

a_test, b_test = 10, 15
predicted_c = predict_sum(a_test, b_test)
print(f"Predicted c for a={a_test}, b={b_test}: {predicted_c:.3f}")
print(f"True sum would be: {a_test + b_test}")
