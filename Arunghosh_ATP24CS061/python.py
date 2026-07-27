import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_data():
    data = pd.read_csv("sum_dataset.csv")
    return data


def build_model(data):
    X = data[["A", "B"]]
    Y = data["C"]

    lr = LinearRegression()
    lr.fit(X, Y)

    return lr


def get_prediction(model):
    num1 = float(input("Enter value of A: "))
    num2 = float(input("Enter value of B: "))

    new_input = pd.DataFrame([[num1, num2]], columns=["A", "B"])

    predicted_value = model.predict(new_input)[0]

    print("\nPredicted Value =", round(predicted_value, 2))

    return num1, num2, predicted_value


def display_graph(data, a, b, predicted):
    plt.figure(figsize=(8, 5))

    x_axis = data["A"] + data["B"]

    plt.scatter(
        x_axis,
        data["C"],
        color="blue",
        label="Dataset"
    )

    plt.scatter(
        a + b,
        predicted,
        color="orange",
        marker="*",
        s=180,
        label="Predicted Point"
    )

    plt.title("Prediction using Linear Regression")
    plt.xlabel("A + B")
    plt.ylabel("C")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()


def run():
    dataset = read_data()
    model = build_model(dataset)

    a, b, result = get_prediction(model)

    display_graph(dataset, a, b, result)


run()
