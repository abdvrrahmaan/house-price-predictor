import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print(df.head())


df = df.dropna(subset=["SalePrice"])
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("Missing")
df = pd.get_dummies(df, drop_first=True)

target_column = "SalePrice"

if target_column not in df.columns:
    raise Exception(f"Target column '{target_column}' not found in dataset")

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Prediction:", prediction[0])
