"""
Статистический анализ образовательных данных с Ассистента преподавателя
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # ✅ Добавлен импорт
from IPython.display import display
import warnings

# Активация стилей Seaborn ✅ Замена plt.style.use('seaborn')
sns.set()

# Настройки отображения
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Загрузка и предварительная очистка данных
    
    Аргументы:
        filepath (str): Путь к файлу Excel
        
    Возвращает:
        pd.DataFrame: Очищенный датафрейм
    """
    try:
        # Загрузка данных
        df = pd.read_excel(filepath, sheet_name="Sheet0")
        
        # Очистка названий колонок
        df.columns = df.columns.str.replace(r'[\n\r\t]', '_', regex=True).str.strip()
        
        # Удаление дублирующих строк
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Преобразование числовых колонок
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    except FileNotFoundError:
        print(f"Ошибка: Файл {filepath} не найден")
        return None

def descriptive_analysis(df):
    """
    Выполняет описательную статистику
    
    Аргументы:
        df (pd.DataFrame): Исходный датафрейм
    """
    print("=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===")
    display(df.describe(include='all'))
    
    # Распределение предметов
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='предмет', order=df['предмет'].value_counts().index)
    plt.title('Распределение предметов')
    plt.xlabel('Количество уроков')
    plt.ylabel('Предмет')
    plt.tight_layout()
    plt.savefig('plots/subject_distribution.png')
    plt.show()

def correlation_analysis(df):
    """
    Корреляционный анализ числовых переменных
    
    Аргументы:
        df (pd.DataFrame): Исходный датафрейм
    """
    print("=== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ===")
    numeric_df = df.select_dtypes(include=['number'])
    
    # Матрица корреляций
    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.show()

def regression_analysis(df):
    """
    Регрессионный анализ
    
    Аргументы:
        df (pd.DataFrame): Исходный датафрейм
    """
    print("=== РЕГРЕССИОННЫЙ АНАЛИЗ ===")
    
    # Подготовка данных
    X = df.drop(['Речь преподавателя', 'предмет', 'параллель'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df['Речь преподавателя']
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Модель случайного леса
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Важность признаков
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='skyblue')
    plt.title('Топ-10 важных признаков')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.show()

def cluster_analysis(df):
    """
    Кластеризация уроков по эмоциональным метрикам
    
    Аргументы:
        df (pd.DataFrame): Исходный датафрейм
    """
    print("=== КЛАСТЕРИЗАЦИЯ ===")
    
    # Подготовка данных
    emotion_cols = ['Позитивная эмоциональная модальность', 
                    'Нейтральная эмоциональная модальность',
                    'Негативная эмоциональная модальность']
    X = df[emotion_cols]
    X = StandardScaler().fit_transform(X)
    
    # Оптимальное количество кластеров
    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 10), inertias, marker='o')
    plt.title('Метод локтя для определения оптимального количества кластеров')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция')
    plt.tight_layout()
    plt.savefig('plots/elbow_method.png')
    plt.show()
    
    # Кластеризация
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Позитивная эмоциональная модальность', 
                    y='Негативная эмоциональная модальность',
                    hue='cluster', data=df, palette='viridis')
    plt.title('Кластеры уроков по эмоциональным метрикам')
    plt.tight_layout()
    plt.savefig('plots/clusters.png')
    plt.show()

def hypothesis_testing(df):
    """
    Проверка гипотез
    
    Аргументы:
        df (pd.DataFrame): Исходный датафрейм
    """
    print("=== ПРОВЕРКА ГИПОТЕЗ ===")
    
    # Гипотеза: Время речи преподавателя различается по предметам
    groups = [group['Речь преподавателя'].values for name, group in df.groupby('предмет')]
    
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("Отвергаем нулевую гипотезу: есть значимые различия между предметами")
    else:
        print("Не можем отвергнуть нулевую гипотезу: различий нет")

# Основной поток выполнения
if __name__ == "__main__":
    # Создаем директорию для графиков
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Загрузка данных
    df = load_data("ОбезличенныеАУ.xlsx")
    if df is not None:
        # Анализ данных
        descriptive_analysis(df)
        correlation_analysis(df)
        regression_analysis(df)
        cluster_analysis(df)
        hypothesis_testing(df)
