import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("AutoInsurSweden.csv", sep = '\t')
df['Y'] = [x.replace(',', '.') for x in df['Y']]
df['Y'] = df['Y'].astype(float)
print(df.head())
X = df['X']
Y = df['Y']
X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size = 0.15))

plt.scatter(X_train, Y_train)
plt.xlabel("X_train")
plt.ylabel("Y_train")
plt.show()

print("X_train contain = ", X_train.shape, "    and    Y_train contain = ", Y_train.shape)
print("X_test  contain = ", X_test.shape, "    and    Y_test   contain = ", Y_test.shape)

model = LinearRegression()
model.fit(X_train.values.reshape(-1,1), Y_train.values.reshape(-1,1))
#prediction = model.predict(X_test.values.reshape(-1,1))

score = model.score(X_test.values.reshape(-1,1), Y_test.values.reshape(-1,1))

print("Score = ", score)