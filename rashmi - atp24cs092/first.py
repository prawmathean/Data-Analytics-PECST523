import pandas as pd
data = {
    "Num1": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    "Num2": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    "Sum":  [4,7,9,10,11,13,14,17,18,21,23,25,28,29,31,33,35,37,40,42,43,45,47,52,51,53,55,57,59,61]
}
df = pd.DataFrame(data)
print(df)
import matplotlib.pyplot as plt
plt.xlabel('First Number')
plt.ylabel('Sum')
plt.scatter(df.Num1,df.Sum,color = 'red')
plt.show()
X = df[['Num1','Num2']]
y = df[['Sum']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
len(X_train)
len(X_test)
from sklearn.linear_model import LinearRegression
c = LinearRegression()
c.fit(X_train,y_train)
c.predict(X_test)
y_test
plt.scatter(X_test['Num1'], y_test, color='blue', label='Actual Sum')
plt.scatter(X_test['Num1'], c.predict(X_test), color='green', label='Predicted Sum')
plt.xlabel('First Number')
plt.ylabel('Sum')
plt.title('Actual vs. Predicted Sum based on First Number')
plt.legend()
plt.show()
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))
new_data = pd.DataFrame([[num1, num2]], columns=['Num1', 'Num2'])
predicted_sum = c.predict(new_data)
print(predicted_sum[0][0])