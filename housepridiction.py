
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
dataset = pd.read_csv(
    "/Users/dhruvchhatrawal/Desktop/applied/Housing.csv"
)

# 2. Separate input and output
# Price is the target column
X = dataset.drop("price", axis=1)
Y = dataset["price"]

# 3. Convert categorical data into numerical data
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 4. First Split (1/3 testing)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

# 5. Train Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# 6. Predict values
y_pred_lin = lin_reg.predict(X_test)

# 7. Evaluate First Model
print("=== First Split (1/3) ===")
print("Linear Regression Weight:", lin_reg.coef_)
print("Linear Regression Bias:", lin_reg.intercept_)

print("MSE:", mean_squared_error(Y_test, y_pred_lin))
print("MAE:", mean_absolute_error(Y_test, y_pred_lin))
print("R2 Score:", r2_score(Y_test, y_pred_lin))

# 8. First Model Graph
plt.figure(figsize=(8, 5))

plt.scatter(Y_test, y_pred_lin, color="green", label="Predicted Prices")

plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    color="red",
    label="Perfect Prediction"
)

plt.title("Dhruv's House Price Prediction - Split 1/3")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# SECOND MODEL: 1/4 TESTING DATA
# ==========================================

# 9. Second Train-Test Split
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    X, Y, test_size=1/4, random_state=0
)

# 10. Train Second Model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_train2, Y_train2)

# 11. Predict values
y_pred_lin2 = lin_reg2.predict(X_test2)

# 12. Evaluate Second Model
print("\n=== Second Split (1/4) ===")
print("Linear Regression Weight:", lin_reg2.coef_)
print("Linear Regression Bias:", lin_reg2.intercept_)

print("MSE:", mean_squared_error(Y_test2, y_pred_lin2))
print("MAE:", mean_absolute_error(Y_test2, y_pred_lin2))
print("R2 Score:", r2_score(Y_test2, y_pred_lin2))

# 13. Second Model Graph
plt.figure(figsize=(8, 5))

plt.scatter(Y_test2, y_pred_lin2, color="blue", label="Predicted Prices")

plt.plot(
    [Y_test2.min(), Y_test2.max()],
    [Y_test2.min(), Y_test2.max()],
    color="red",
    label="Perfect Prediction"
)

plt.title("Dhruv's House Price Prediction - Split 1/4")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()

print("\nHouse Price Prediction Model Executed Successfully!")
