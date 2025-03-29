import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Генерация данных с зависимостью
np.random.seed(42)  # Для воспроизводимости

# Категории посещаемости
categories = [
    'не знал (не знала), что практические занятия проводятся',
    'не посещал (не посещала)',
    'иногда посещал (посещала) и было не интересно',
    'посещал (посещала) не всегда, занятия были интересны',
    'никогда не пропускал, на занятиях было интересно'
]

# Создание датафрейма
data = {
    'X': categories * 60,
    'Y': []  # Пустой список для оценок
}

# Генерация оценок с зависимостью от категории
for category in categories:
    if category == categories[0]:  # "не знал"
        data['Y'].extend(np.random.randint(0, 2, size=60))  # Низкие оценки
    elif category == categories[1]:  # "не посещал"
        data['Y'].extend(np.random.randint(0, 3, size=60))  # Средние низкие оценки
    elif category == categories[2]:  # "иногда посещал"
        data['Y'].extend(np.random.randint(1, 4, size=60))  # Средние оценки
    elif category == categories[3]:  # "посещал не всегда"
        data['Y'].extend(np.random.randint(2, 5, size=60))  # Средние высокие оценки
    elif category == categories[4]:  # "никогда не пропускал"
        data['Y'].extend(np.random.randint(3, 6, size=60))  # Высокие оценки

df = pd.DataFrame(data)

# Преобразование X в порядковую шкалу
order_mapping = {
    'не знал (не знала), что практические занятия проводятся': 1,
    'не посещал (не посещала)': 2,
    'иногда посещал (посещала) и было не интересно': 3,
    'посещал (посещала) не всегда, занятия были интересны': 4,
    'никогда не пропускал, на занятиях было интересно': 5
}
df['X_Ordered'] = df['X'].map(order_mapping)

# Корреляционный анализ с использованием коэффициента Спирмана
rho, p_value = stats.spearmanr(df['X_Ordered'], df['Y'])
print(f"Коэффициент Спирмана (ρ): {rho}")
print(f"P-значение: {p_value}")

# Интерпретация:
if p_value < 0.05:
    print("Связь между переменными статистически значима.")
else:
    print("Связь между переменными не является статистически значимой.")

# Визуализация: Диаграмма рассеяния
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X_Ordered', y='Y', data=df)
plt.title('Диаграмма рассеяния: Посещаемость занятий vs Впечатление от программы')
plt.xlabel('Посещаемость занятий (порядковая шкала)')
plt.ylabel('Впечатление от программы (оценка от 0 до 5)')
plt.show()

# Визуализация: Бокс-плот
plt.figure(figsize=(10, 6))
sns.boxplot(x='X', y='Y', data=df)
plt.title('Распределение впечатлений от программы по категориям посещаемости занятий')
plt.xticks(rotation=90)
plt.show()