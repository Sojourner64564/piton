import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_wine_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    wine_data = pd.read_csv(url, delimiter=';')
    wine_data['quality_category'] = pd.cut(wine_data['quality'],
                                         bins=[0, 4, 6, 10],
                                         labels=['Низкое', 'Среднее', 'Высокое'])
    return wine_data

# Загрузка данных
wine_data = load_wine_data()

# 1. Исследование характеристик

# 1.1 Распределение показателей качества вин
plt.figure(figsize=(10, 6))
quality_counts = wine_data['quality_category'].value_counts().sort_index()
plt.bar(quality_counts.index, quality_counts.values)
plt.title('Распределение вин по категориям качества')
plt.xlabel('Категория качества')
plt.ylabel('Количество образцов')
plt.show()

# 1.2 Анализ выбросов в химических показателях
chemical_columns = wine_data.columns[:-2]  # исключаем quality и quality_category
plt.figure(figsize=(15, 10))
wine_data[chemical_columns].boxplot()
plt.title('Выбросы в химических показателях вина')
plt.xticks(rotation=45)
plt.ylabel('Значения показателей')
plt.tight_layout()
plt.show()

# 1.3 Изучение корреляций между свойствами вина
plt.figure(figsize=(12, 10))
correlation_matrix = wine_data[chemical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляций между химическими показателями вина')
plt.tight_layout()
plt.show()

# 2. Сравнительный анализ

# 2.1 Сравнение химического состава вин разного качества
chemical_features = ['fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'alcohol']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(chemical_features):
    wine_data.boxplot(column=feature, by='quality_category', ax=axes[i])
    axes[i].set_title(f'Распределение {feature} по качеству')
    axes[i].set_xlabel('Категория качества')

plt.suptitle('')
plt.tight_layout()
plt.show()

# 2.2 Влияние кислотности на общую оценку
acidity_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'pH']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(acidity_features):
    for quality_cat in wine_data['quality_category'].unique():
        subset = wine_data[wine_data['quality_category'] == quality_cat]
        axes[i].hist(subset[feature], alpha=0.7, label=quality_cat, bins=15)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Частота')
    axes[i].set_title(f'Распределение {feature} по качеству')
    axes[i].legend()

plt.tight_layout()
plt.show()

# 2.3 Анализ связи алкоголя и качества
plt.figure(figsize=(10, 6))
sns.boxplot(data=wine_data, x='quality_category', y='alcohol')
plt.title('Содержание алкоголя в винах разного качества')
plt.xlabel('Категория качества')
plt.ylabel('Содержание алкоголя (%)')
plt.show()

# 3. Гипотезы и проверки

# 3.1 Влияние уровня сахара на воспринимаемое качество
plt.figure(figsize=(10, 6))
sns.scatterplot(data=wine_data, x='residual sugar', y='quality',
                hue='quality_category', alpha=0.7)
plt.title('Влияние остаточного сахара на качество вина')
plt.xlabel('Остаточный сахар (г/л)')
plt.ylabel('Качество')
plt.legend(title='Категория качества')
plt.show()

# 3.2 Связь между pH и кислотностью
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Связь pH с фиксированной кислотностью
sns.scatterplot(data=wine_data, x='fixed acidity', y='pH',
                hue='quality_category', ax=ax1, alpha=0.7)
ax1.set_title('Связь pH с фиксированной кислотностью')
ax1.set_xlabel('Фиксированная кислотность')
ax1.set_ylabel('pH')

# Связь pH с летучей кислотностью
sns.scatterplot(data=wine_data, x='volatile acidity', y='pH',
                hue='quality_category', ax=ax2, alpha=0.7)
ax2.set_title('Связь pH с летучей кислотностью')
ax2.set_xlabel('Летучая кислотность')
ax2.set_ylabel('pH')

plt.tight_layout()
plt.show()

# 3.3 Статистическая проверка различий между группами качества
quality_groups = {}
for category in wine_data['quality_category'].unique():
    quality_groups[category] = wine_data[wine_data['quality_category'] == category]['alcohol']

# Проверка нормальности распределения
normality_results = {}
for category, group_data in quality_groups.items():
    stat, p_value = stats.normaltest(group_data)
    normality_results[category] = p_value

print("Проверка нормальности распределения алкоголя:")
for category, p_value in normality_results.items():
    print(f"{category}: p-value = {p_value:.4f}")

# ANOVA тест для сравнения средних
f_stat, p_value = stats.f_oneway(*quality_groups.values())
print(f"\nANOVA тест для содержания алкоголя:")
print(f"F-статистика: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Post-hoc тест Тьюки для попарного сравнения
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_results = pairwise_tukeyhsd(
    wine_data['alcohol'].dropna(),
    wine_data['quality_category'].dropna(),
    alpha=0.05
)
print(f"\nРезультаты теста Тьюки:")
print(tukey_results)