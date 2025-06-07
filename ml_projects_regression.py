import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Example dataset: x = hours studied, y = test score
x = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
y = np.array([81, 93, 91, 97, 109])

# For smooth curves
x_plot = np.linspace(2, 10, 100).reshape(-1, 1)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_lin_pred = lin_reg.predict(x_plot)

# 2. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)
y_poly_pred = poly_reg.predict(poly.transform(x_plot))

# 3. Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(x, y)
y_rf_pred = rf_reg.predict(x_plot)

# 4. K-Nearest Neighbors Regression (k=2)
knn_reg = KNeighborsRegressor(n_neighbors=2)
knn_reg.fit(x, y)
y_knn_pred = knn_reg.predict(x_plot)

# ---- USER INPUT ----
user_x = float(input("Enter hours studied (between 2 and 10): "))
user_x_arr = np.array([[user_x]])

# Predict with all models
user_pred_lin = lin_reg.predict(user_x_arr)[0]
user_pred_poly = poly_reg.predict(poly.transform(user_x_arr))[0]
user_pred_rf = rf_reg.predict(user_x_arr)[0]
user_pred_knn = knn_reg.predict(user_x_arr)[0]

print(f"\nPredictions for {user_x} hours studied:")
print(f"Linear Regression:        {user_pred_lin:.2f}")
print(f"Polynomial Regression:    {user_pred_poly:.2f}")
print(f"Random Forest Regression: {user_pred_rf:.2f}")
print(f"KNN Regression:           {user_pred_knn:.2f}")

# ---- PLOTTING ----
plt.figure(figsize=(10, 7))
plt.scatter(x, y, color='black', label='Data Points', s=60)
plt.plot(x_plot, y_lin_pred, color='blue', label='Linear Regression')
plt.plot(x_plot, y_poly_pred, color='red', label='Polynomial Regression (deg 2)')
plt.plot(x_plot, y_rf_pred, color='green', label='Random Forest Regression')
plt.plot(x_plot, y_knn_pred, color='purple', label='KNN Regression (k=2)')

# Plot user prediction points
plt.scatter([user_x], [user_pred_lin], color='blue', s=100, marker='x')
plt.scatter([user_x], [user_pred_poly], color='red', s=100, marker='x')
plt.scatter([user_x], [user_pred_rf], color='green', s=100, marker='x')
plt.scatter([user_x], [user_pred_knn], color='purple', s=100, marker='x')

plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Regression Algorithm Comparison')
plt.legend()
plt.show()
