import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 1. Загрузка датасета
df = sns.load_dataset('titanic')

# 2. Анализ данных с помощью pandas

# 2.1 Анализ структуры данных
print("=== АНАЛИЗ СТРУКТУРЫ ДАННЫХ ===")
print("Информация о датафрейме:")
print(df.info())
print("\nПропущенные значения:")
print(df.isnull().sum())

# 2.2 Статистическое описание числовых признаков
print("\n=== СТАТИСТИЧЕСКОЕ ОПИСАНИЕ ===")
print(df.describe())

# 2.3 Группировка данных по полу и классу каюты с расчетом выживаемости
print("\n=== ВЫЖИВАЕМОСТЬ ПО ПОЛУ И КЛАССУ ===")
survival_by_sex_class = df.groupby(['sex', 'class'], observed=True)['survived'].agg(['mean', 'count']).round(3)
#survival_by_sex_class = df.groupby(['sex', 'class'])['survived'].agg(['mean', 'count']).round(3)
print(survival_by_sex_class)


# 2.4 Создание новых признаков
def create_features(df):
    # Добавляем возрастные группы
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 18, 30, 50, 100],
                             labels=['child', 'young', 'adult', 'senior'])

    df['family_size'] = df['sibsp'] + df['parch'] # Добавим размер семьи
    return df


df = create_features(df)

# 3. Сравнение производительности с разными типами данных

# 3.1 Стандартные типы pandas
print("\n=== СТАНДАРТНЫЕ ТИПЫ PANDAS ===")
print(f"Используемая память: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# 3.2 Категориальные типы
df_category = df.copy()
categorical_columns = ['sex', 'class', 'embarked', 'who', 'deck', 'embark_town', 'alone', 'age_group']
for col in categorical_columns:
    if col in df_category.columns:
        df_category[col] = df_category[col].astype('category')

print("\n=== КАТЕГОРИАЛЬНЫЕ ТИПЫ ===")
print(f"Используемая память: {df_category.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# 3.3 Типы PyArrow
try:
    df_pyarrow = df.copy()
    # Конвертируем строковые колонки в PyArrow string
    string_columns = df_pyarrow.select_dtypes(include=['object']).columns
    for col in string_columns:
        df_pyarrow[col] = df_pyarrow[col].astype('string[pyarrow]')

    print("\n=== ТИПЫ PYARROW ===")
    print(f"Используемая память: {df_pyarrow.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
except ImportError:
    print("PyArrow не установлен")

# 4. Визуализация с seaborn

# Обработка пропущенных значений для визуализации
df_viz = df.dropna(subset=['age', 'fare']).copy()

# 4.1 Матрица корреляций
plt.figure(figsize=(12, 10))
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
correlation_matrix = df_viz[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Матрица корреляций числовых признаков Titanic', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 4.2 Распределение возрастов с разбивкой по полу и выживаемости
plt.figure(figsize=(12, 8))
sns.histplot(data=df_viz, x='age', hue='survived', multiple='stack',
             palette={0: 'red', 1: 'green'}, alpha=0.7, kde=True)
plt.title('Распределение возрастов пассажиров по выживаемости', fontsize=14)
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.legend(['Погиб', 'Выжил'], title='Выживаемость')
plt.grid(True, alpha=0.3)
plt.show()

# 4.3 Количество выживших по классу каюты и порту посадки
plt.figure(figsize=(12, 8))
#survival_count = df_viz.groupby(['class', 'embark_town', 'survived']).size().unstack()
survival_count = df_viz.groupby(['class', 'embark_town', 'survived'], observed=True).size().unstack()
survival_count.plot(kind='bar', stacked=True, color=['red', 'green'], alpha=0.8)
plt.title('Количество выживших по классу каюты и порту посадки', fontsize=14)
plt.xlabel('Класс каюты - Порт посадки')
plt.ylabel('Количество пассажиров')
plt.legend(['Погиб', 'Выжил'], title='Выживаемость')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4.4 Ящик с усами стоимости билета по классам
plt.figure(figsize=(12, 8))
#sns.boxplot(data=df_viz, x='class', y='fare', palette='Set2')
sns.boxplot(data=df_viz, x='class', y='fare', hue='class', palette='Set2', legend=False)
plt.title('Распределение стоимости билетов по классам каюты', fontsize=14)
plt.xlabel('Класс каюты')
plt.ylabel('Стоимость билета')
plt.grid(True, alpha=0.3)
plt.show()


# 5. Интерактивная визуализация с фильтрацией
def create_interactive_dashboard(df):
    fig = plt.figure(figsize=(15, 10))

    # Параметры фильтрации
    age_filter = (18, 60)
    class_filter = [1, 2, 3]

    # Применяем фильтры
    filtered_df = df[
        (df['age'].between(*age_filter)) &
        (df['pclass'].isin(class_filter))
        ].dropna(subset=['age', 'fare'])

    # Создаем сетку для графиков
    gs = GridSpec(2, 2, figure=fig)

    # График 1: Распределение возрастов по классам
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=filtered_df, x='age', hue='class', multiple='stack',
                 palette='Set2', alpha=0.7, ax=ax1)
    ax1.set_title('Распределение возрастов по классам каюты\n(фильтр: 18-60 лет)', fontsize=12)
    ax1.set_xlabel('Возраст')
    ax1.set_ylabel('Количество пассажиров')
    ax1.grid(True, alpha=0.3)

    # График 2: Выживаемость по классам и полу
    ax2 = fig.add_subplot(gs[0, 1])
    #survival_rate = filtered_df.groupby(['class', 'sex'])['survived'].mean().unstack()
    survival_rate = filtered_df.groupby(['class', 'sex'], observed=True)['survived'].mean().unstack()
    survival_rate.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'], alpha=0.8)
    ax2.set_title('Процент выживших по классам и полу\n(фильтр: 18-60 лет)', fontsize=12)
    ax2.set_xlabel('Класс каюты')
    ax2.set_ylabel('Доля выживших')
    ax2.legend(title='Пол')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # График 3: Стоимость билета vs Возраст с разбивкой по выживаемости
    ax3 = fig.add_subplot(gs[1, :])
    scatter = ax3.scatter(filtered_df['age'], filtered_df['fare'],
                          c=filtered_df['survived'], cmap='RdYlGn',
                          alpha=0.6, s=50)
    ax3.set_title('Стоимость билета vs Возраст с разбивкой по выживаемости\n(фильтр: 18-60 лет)', fontsize=12)
    ax3.set_xlabel('Возраст')
    ax3.set_ylabel('Стоимость билета')
    ax3.grid(True, alpha=0.3)

    # Добавляем цветовую легенду
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Выживаемость')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Погиб', 'Выжил'])

    plt.tight_layout()
    return fig


# Создаем интерактивную витрину
dashboard = create_interactive_dashboard(df)
plt.show()

# Дополнительный анализ: обработка пропущенных значений
print("\n=== ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
print(f"Исходный размер датасета: {len(df)}")
print(f"Пропуски в возрасте: {df['age'].isnull().sum()} ({df['age'].isnull().mean() * 100:.1f}%)")
print(f"Пропуски в порте посадки: {df['embarked'].isnull().sum()} ({df['embarked'].isnull().mean() * 100:.1f}%)")

# Заполняем пропуски в возрасте медианным значением по классу и полу
df_cleaned = df.copy()
df_cleaned['age'] = df_cleaned.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)

# Заполняем пропуски в порте посадки модой
df_cleaned['embarked'] = df_cleaned['embarked'].fillna(df_cleaned['embarked'].mode()[0])

print(f"Размер после обработки пропусков: {len(df_cleaned)}")
print(f"Оставшиеся пропуски в возрасте: {df_cleaned['age'].isnull().sum()}")