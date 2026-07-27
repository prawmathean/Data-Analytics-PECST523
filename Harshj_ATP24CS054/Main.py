import pandas as pd

# ---- Load data ----
dataframe = pd.read_csv("train_data.csv")

x = dataframe[["X", "Y"]].values.astype(float)
y = dataframe["sum"].values.astype(float)  # this is the actual (mostly correct) value

# ---- Model ----
# prediction = x*w1 + y*w2 + c
#   x, y   - the actual values in the row
#   w1, w2 - learned weights; once trained, w1 -> 1, w2 -> 1, c -> 0
#            so the model effectively becomes prediction = x + y

w1 = 0.0
w2 = 0.0
c = 0.0
lr = 0.1          # learning rate
scale = 1000.0    # keeps inputs in a small range (~0-1) for stable gradient descent
epoch_range = 100


def predict(a, b, w1, w2, c, scale):
    """Predict x+y for raw (unscaled) inputs a, b using the current weights."""
    a_scaled = a / scale
    b_scaled = b / scale
    predicted_scaled = a_scaled * w1 + b_scaled * w2 + c
    return predicted_scaled * scale


def ask_model(a, b):
    result = predict(a, b, w1, w2, c, scale)
    print(f"Model: {a} + {b} = {result:.2f}  (actual: {a + b})")


# ---- Training loop ----
for epoch in range(epoch_range):
    last_error = 0.0
    for i in range(len(x)):
        a = x[i][0] / scale         # first column value of i-th row, scaled
        b = x[i][1] / scale         # second column value of i-th row, scaled
        actual_sum = y[i] / scale   # the true x+y, scaled

        prediction = a * w1 + b * w2 + c
        error = prediction - actual_sum
        last_error = error

        # Gradient descent step:
        # if prediction is too high (error > 0), nudge weights down;
        # if prediction is too low (error < 0), nudge weights up.
        w1 = w1 - lr * error * a
        w2 = w2 - lr * error * b
        c = c - lr * error

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, w1={w1:.4f}, w2={w2:.4f}, c={c:.4f}, last_error={last_error:.4f}", end="\r")

print(f"\nModel trained with {len(x)} rows over {epoch_range} epochs.")
print(f"Final weights -> w1={w1:.4f}, w2={w2:.4f}, c={c:.4f}\n")

# ---- Test ----
ask_model(3, 4)
ask_model(120, 380)
ask_model(999, 1)
ask_model(500, 500)
