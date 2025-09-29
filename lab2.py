import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

X = np.array([12.000000000000014]).reshape(-1, 1)
y = np.array([1.7643614775905017])

print("Початкові дані:")
print(f"X: {X.flatten()}")
print(f"y: {y}")


np.random.seed(42)
X_extended = np.linspace(10, 14, 20).reshape(-1, 1)
y_extended = 0.1 * (X_extended - 12)**2 + 1.76 + np.random.normal(0, 0.05, X_extended.shape)


X_train, X_test, y_train, y_test = train_test_split(X_extended, y_extended, test_size=0.3, random_state=42)

print(f"\nРозмірність даних:")
print(f"Навчальна вибірка: {X_train.shape[0]} спостережень")
print(f"Тестова вибірка: {X_test.shape[0]} спостережень")

# 1. Лінійна регресійна модель
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 2. Поліноміальні регресійні моделі
degrees = [2, 3, 6, 12]
poly_models = {}
poly_scores = {}

for degree in degrees:
    # Створення поліноміальних ознак
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)
    
    # Навчання моделі
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    
    # Прогнози
    y_train_pred = poly_model.predict(X_poly_train)
    y_test_pred = poly_model.predict(X_poly_test)
    
    # Збереження моделі та результатів
    poly_models[degree] = {
        'model': poly_model,
        'features': poly_features,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred
    }
    
    # Обчислення метрик
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    poly_scores[degree] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse
    }

# 3. Порівняння показників
print("\n" + "="*80)
print("ПОРІВНЯННЯ МОДЕЛЕЙ")
print("="*80)

# Лінійна модель
y_train_linear = linear_model.predict(X_train)
y_test_linear = linear_model.predict(X_test)

linear_scores = {
    'train_r2': r2_score(y_train, y_train_linear),
    'test_r2': r2_score(y_test, y_test_linear),
    'train_mse': mean_squared_error(y_train, y_train_linear),
    'test_mse': mean_squared_error(y_test, y_test_linear)
}

print(f"\nЛінійна модель:")
print(f"R² на тренувальних даних: {linear_scores['train_r2']:.4f}")
print(f"R² на тестових даних: {linear_scores['test_r2']:.4f}")
print(f"MSE на тренувальних даних: {linear_scores['train_mse']:.6f}")
print(f"MSE на тестових даних: {linear_scores['test_mse']:.6f}")

for degree in degrees:
    scores = poly_scores[degree]
    print(f"\nПоліноміальна модель {degree} ступеня:")
    print(f"R² на тренувальних даних: {scores['train_r2']:.4f}")
    print(f"R² на тестових даних: {scores['test_r2']:.4f}")
    print(f"MSE на тренувальних даних: {scores['train_mse']:.6f}")
    print(f"MSE на тестових даних: {scores['test_mse']:.6f}")

# 4. Побудова графіків
plt.figure(figsize=(15, 10))

# Сортування даних для плавних графіків
sort_idx = np.argsort(X_test.flatten())
X_test_sorted = X_test[sort_idx]
y_test_sorted = y_test[sort_idx]

# Лінійна модель
plt.subplot(2, 3, 1)
plt.scatter(X_train, y_train, alpha=0.7, label='Тренувальні дані', color='blue')
plt.scatter(X_test, y_test, alpha=0.7, label='Тестові дані', color='red')
X_plot = np.linspace(10, 14, 100).reshape(-1, 1)
y_plot_linear = linear_model.predict(X_plot)
plt.plot(X_plot, y_plot_linear, 'g-', linewidth=2, label='Лінійна модель')
plt.title('Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Поліноміальні моделі
for i, degree in enumerate(degrees):
    plt.subplot(2, 3, i+2)
    plt.scatter(X_train, y_train, alpha=0.7, label='Тренувальні дані', color='blue')
    plt.scatter(X_test, y_test, alpha=0.7, label='Тестові дані', color='red')
    
    # Прогноз для графіка
    poly_features = poly_models[degree]['features']
    X_poly_plot = poly_features.transform(X_plot)
    y_plot_poly = poly_models[degree]['model'].predict(X_poly_plot)
    
    plt.plot(X_plot, y_plot_poly, 'orange', linewidth=2, label=f'Поліном {degree} ступеня')
    plt.title(f'Поліноміальна регресія (ступінь {degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Графік порівняння якості моделей
plt.figure(figsize=(12, 5))

# R² порівняння
plt.subplot(1, 2, 1)
degrees_all = ['Linear'] + degrees
train_r2_scores = [linear_scores['train_r2']] + [poly_scores[d]['train_r2'] for d in degrees]
test_r2_scores = [linear_scores['test_r2']] + [poly_scores[d]['test_r2'] for d in degrees]

x_pos = np.arange(len(degrees_all))
width = 0.35

plt.bar(x_pos - width/2, train_r2_scores, width, label='Тренувальні дані', alpha=0.7)
plt.bar(x_pos + width/2, test_r2_scores, width, label='Тестові дані', alpha=0.7)
plt.xlabel('Ступінь полінома')
plt.ylabel('R² score')
plt.title('Порівняння R² для різних моделей')
plt.xticks(x_pos, degrees_all)
plt.legend()
plt.grid(True, alpha=0.3)

# MSE порівняння
plt.subplot(1, 2, 2)
train_mse_scores = [linear_scores['train_mse']] + [poly_scores[d]['train_mse'] for d in degrees]
test_mse_scores = [linear_scores['test_mse']] + [poly_scores[d]['test_mse'] for d in degrees]

plt.bar(x_pos - width/2, train_mse_scores, width, label='Тренувальні дані', alpha=0.7)
plt.bar(x_pos + width/2, test_mse_scores, width, label='Тестові дані', alpha=0.7)
plt.xlabel('Ступінь полінома')
plt.ylabel('MSE')
plt.title('Порівняння MSE для різних моделей')
plt.xticks(x_pos, degrees_all)
plt.legend()
plt.yscale('log')  # Логарифмічна шкала для кращого відображення
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Висновки
print("\n" + "="*80)
print("ВИСНОВКИ")
print("="*80)

# Знаходження оптимального ступеня
best_degree = None
best_test_r2 = -np.inf

for degree in ['Linear'] + degrees:
    if degree == 'Linear':
        test_r2 = linear_scores['test_r2']
    else:
        test_r2 = poly_scores[degree]['test_r2']
    
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_degree = degree

print(f"\nОптимальний ступінь полінома: {best_degree}")
print(f"Найкращий R² на тестових даних: {best_test_r2:.4f}")

print("\nАналіз впливу складності моделі:")
print("1. Лінійна модель: добре узагальнює, але може не захоплювати нелінійні залежності")
print("2. Поліном 2 ступеня: часто оптимальний баланс між складністю та узагальненням")
print("3. Поліном 3 ступеня: може краще відповідати даним, але ризик перенавчання")
print("4. Поліном 6 ступеня: високий ризик перенавчання, особливо при малій кількості даних")
print("5. Поліном 12 ступеня: дуже високий ризик перенавчання, модель стає надто складною")
