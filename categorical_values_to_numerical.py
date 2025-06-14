import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n = 100
size = np.random.rand(n) * 1000
neighborhood = np.random.choice(['Downtown', 'Suburb', 'Rural'], n)
price = 50 * size + np.where(neighborhood == 'Downtown', 20000, 0) + \
        np.where(neighborhood == 'Suburb', 10000, 0) + np.random.randn(n) * 5000

# Create a DataFrame
data = pd.DataFrame({'Size': size, 'Neighborhood': neighborhood, 'Price': price})

# Check for missing values
print("Missing values:\n", data.isna().sum())

# One-hot encode the categorical variable
X = pd.get_dummies(data[['Size', 'Neighborhood']], columns=['Neighborhood'], drop_first=True).astype(np.float64)
print("X dtypes:\n", X.dtypes)

# Add intercept term
X_b = np.c_[np.ones((n, 1)), X]
print("X_b dtype:", X_b.dtype)

# Ensure y is numeric and reshaped
y = data['Price'].astype(np.float64).values.reshape(-1, 1)
print("y dtype:", y.dtype)

# OLS calculation
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Make predictions
y_pred = X_b @ beta

# Calculate R-squared
ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r_squared = 1 - ss_res / ss_tot

# Print results
print(f"Intercept: {beta[0, 0]:.2f}")
print(f"Coefficients: {beta[1:, 0]}")
print(f"R-squared: {r_squared:.4f}")

# Plot
plt.scatter(data['Size'], data['Price'], c=pd.Categorical(data['Neighborhood']).codes, 
            cmap='viridis', label='Data points')
plt.plot(data['Size'], y_pred, color='red', label='OLS fit (approx)')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression with Categorical Variable')
plt.legend()
plt.show()
