import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
num_samples = 100
X = np.random.rand(num_samples, 1) * 100  # suprafata locuintei
y = 5 * X + np.random.randn(num_samples, 1) * 20  # Pretul locuintei

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)


plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression Model')
plt.title("Exemplu regresie liniara in imobiliare")
plt.xlabel("House area (sq. meters")
plt.ylabel('House price')
plt.legend()
plt.show()

r2 = r2_score(y_test, y_pred)
print(f"Coeficientul de interceptare (theta0): {model.intercept_[0]}")
print(f"Coeficientul de inclinare (theta1): {model.coef_[0][0]}")
print(f"Eroarea medie patratica (MSE): {mse}")
print(f"R-squared: {r2}")

