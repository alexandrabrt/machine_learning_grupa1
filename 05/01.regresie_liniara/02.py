import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# generam date fictive
np.random.seed(42)
num_samples = 100
education_year = np.random.randint(8, 20, size=(num_samples, 1))  # numarul de ani de educatie
experience_year = np.random.randint(0, 30, size=(num_samples, 1))  # numarul de ani de experienta
income = 2000 + 100 * education_year + 50 * experience_year + np.random.randn(num_samples, 1) * 500


# impartim datele in set de antrenare si set de test
X = np.hstack((education_year, experience_year))
y = income
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# antrenam modelul
model = LinearRegression()
model.fit(X_train, y_train)

# efectuam predictiile pe setul de testare
y_pred = model.predict(X_test)

# calculam eroarea medie patratica
mse = mean_squared_error(y_test, y_pred)

# vizualizam rezultatele
fig, ax = plt.subplots(1, 2, figsize=(12, 4))


# vizualizam predictiile si datele reale
ax[0].scatter(y_test, y_pred, color='blue')
ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
ax[0].set_xlabel("Venit real")
ax[0].set_ylabel("Venit prezis")
ax[0].set_title("Predictii vs real")


# vizualizam reziduurile
residuals = y_test - y_pred
ax[1].scatter(y_pred, residuals, color='green')
ax[1].axhline(y=0, linestyle='--', color='red', linewidth=2)
ax[1].set_xlabel('Venit prezis')
ax[1].set_ylabel('Reziduuri')
ax[1].set_title("Reziduuri vs predictii")

plt.show()

print(f"Coeficientul de intercepatare (beta0): {model.intercept_}")
print(f"Coeficientii de inclinare (beta1, beta2): {model.coef_}")
print(f"Eroarea medie patratica (MSE): {mse}")
