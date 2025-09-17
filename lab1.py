import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1. Генерація синтетичних даних
# =========================
np.random.seed(42)  # щоб результати відтворювалися
n = 12  # твій варіант
x = np.linspace(0, 10, 100)
noise = np.random.normal(0, 0.3, size=100)  # шум
y = n * x + np.sin(x / n) + noise

# перетворюємо x у стовпчик для sklearn
X = x.reshape(-1, 1)

# =========================
# 2. Розділення на train (70%) і test (30%)
# =========================
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 3. Побудова лінійної регресії
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

a = model.coef_[0]
b = model.intercept_

print(f"Коефіцієнти регресії: a = {a:.4f}, b = {b:.4f}")

# =========================
# 4. Оцінка моделі
# =========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE = {mse:.4f}")
print(f"MAE = {mae:.4f}")
print(f"R^2 = {r2:.4f}")

# =========================
# 5. Графік даних та лінії регресії
# =========================
plt.scatter(X_test, y_test, color='blue', label="Тестові дані")
plt.plot(X_test, y_pred, color='red', label="Лінія регресії")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Лінійна регресія (варіант 12)")
plt.legend()
plt.show()

# =========================
# 6. Висновок
# =========================
if r2 > 0.9:
    conclusion = "Модель добре описує дані."
elif r2 > 0.7:
    conclusion = "Модель середньої якості, придатна для приблизних прогнозів."
else:
    conclusion = "Модель слабко описує дані, потрібні складніші методи."

print("Висновок:", conclusion)
