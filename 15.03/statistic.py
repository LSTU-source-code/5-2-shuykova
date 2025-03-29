import pandas as pd

# Загрузка данных
file_path = 'МониторингОбразование)_2025_Обезличенный.xlsx - Таблица 1.csv'
data = pd.read_csv(file_path, header=None)

# Первые строки таблицы
print("Первые строки данных:")
print(data.head())

# Информация о данных
print("\nИнформация о данных:")
print(data.info())

# Преобразование числовых столбцов
# Пробуем преобразовать все столбцы, кроме первого (заголовки)
for col in data.columns[1:]:
    # Преобразуем в числовой формат, игнорируя ошибки
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Описательная статистика для числовых столбцов
numeric_stats = data.describe(include='number')
print("\nОписательная статистика для числовых столбцов:")
print(numeric_stats)

# Описательная статистика для категориальных столбцов
categorical_stats = data.describe(include='object')
print("\nОписательная статистика для категориальных столбцов:")
print(categorical_stats)

# Количество пропущенных значений по столбцам
missing_values = data.isnull().sum()
print("\nКоличество пропущенных значений по столбцам:")
print(missing_values)

# Анализ по строкам (например, сумма значений в каждой строке)
row_sums = data.sum(axis=1, numeric_only=True)
print("\nСумма значений в каждой строке:")
print(row_sums)

# Анализ по столбцам (например, среднее значение в каждом столбце)
column_means = data.mean(axis=0, numeric_only=True)
print("\nСреднее значение в каждом столбце:")
print(column_means)