import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("addition_dataset_200.csv")
print(df.head(10))

X = df[["N1", "N2"]]
y = df["Sum"]

model = LinearRegression()
model.fit(X, y)

n1 = int(input("Enter first number: "))
n2 = int(input("Enter second number: "))

result = model.predict([[n1, n2]])

print("Predicted Sum:", round(result[0], 2))
print("Actual Sum:", n1 + n2)
