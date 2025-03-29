import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Искусственное создание данных (замените на реальные данные)
data = {
    'X': [
        'было все равно, что изучать',
        'записался по рекомендации учителя',
        'записался за компанию с друзьями',
        'записался по рекомендации друзей',
        'записался по рекомендации родителей',
        'записался по собственному желанию'
    ] * 50,
    'Y': np.random.randint(0, 6, size=300)  # Оценки от 0 до 5
}

df = pd.DataFrame(data)

# Вывод первых строк DataFrame
print(df.head())

# 1. Анализ средних значений Y для каждой категории X
mean_analysis = df.groupby('X')['Y'].mean().reset_index()
print("\nСредние значения Y по категориям X:")
print(mean_analysis)

# 2. Визуализация распределения Y по категориям X
plt.figure(figsize=(10, 6))
sns.boxplot(x='X', y='Y', data=df)
plt.title('Распределение оценок Y по причинам выбора программы X')
plt.xticks(rotation=45)
plt.show()

# 3. Проверка статистической значимости различий между группами
# Пример: Сравнение двух групп (например, "было все равно" и "по рекомендации учителя")
group1_name = 'было все равно, что изучать'
group2_name = 'записался по рекомендации учителя'

group1 = df[df['X'] == group1_name]['Y']
group2 = df[df['X'] == group2_name]['Y']

t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
print(f"\nT-тест между двумя группами ('{group1_name}' vs '{group2_name}'):")
print(f"T-статистика: {t_stat}")
print(f"P-значение: {p_value}")

# Интерпретация результатов T-теста
if p_value < 0.05:
    print("Различия между группами статистически значимы.")
else:
    print("Различия между группами не являются статистически значимыми.")

# 4. Визуализация сравнения двух групп
plt.figure(figsize=(8, 6))
sns.histplot(group1, kde=True, color='blue', label=f'{group1_name} (среднее={group1.mean():.2f})')
sns.histplot(group2, kde=True, color='orange', label=f'{group2_name} (среднее={group2.mean():.2f})')
plt.title(f'Сравнение распределений "{group1_name}" и "{group2_name}"')
plt.xlabel('Оценка (Y)')
plt.ylabel('Частота')
plt.legend()
plt.show()

# 5. ANOVA для всех групп
f_stat, p_value_anova = stats.f_oneway(
    df[df['X'] == 'было все равно, что изучать']['Y'],
    df[df['X'] == 'записался по рекомендации учителя']['Y'],
    df[df['X'] == 'записался за компанию с друзьями']['Y'],
    df[df['X'] == 'записался по рекомендации друзей']['Y'],
    df[df['X'] == 'записался по рекомендации родителей']['Y'],
    df[df['X'] == 'записался по собственному желанию']['Y']
)
print("\nANOVA для всех групп:")
print(f"F-статистика: {f_stat}")
print(f"P-значение: {p_value_anova}")

# Интерпретация результатов ANOVA
if p_value_anova < 0.05:
    print("Различия между группами статистически значимы.")
else:
    print("Различия между группами не являются статистически значимыми.")