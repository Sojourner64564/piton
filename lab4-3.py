import pandas as pd
import numpy as np
import polars as pl
import pyarrow as pa
import time
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage


def generate_large_dataset(n_rows=1000000):
    return pd.DataFrame({
        'id': range(n_rows),
        'timestamp': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value1': np.random.normal(0, 1, n_rows),
        'value2': np.random.exponential(1, n_rows),
        'value3': np.random.randint(0, 100, n_rows)
    })


# Операция 1: Фильтрация и агрегация
def pandas_filter_aggregate(df):
    result = (df[df['value1'] > 0]
              .groupby('category')
              .agg({'value2': ['mean', 'std'], 'value1': 'count'})
              .round(4))
    result.columns = ['avg_value', 'std_value', 'count']
    return result.sort_values('avg_value', ascending=False)


def pandas_pyarrow_filter_aggregate(df):
    df_pa = df.astype({
        'category': 'string[pyarrow]',
        'value1': 'float64[pyarrow]',
        'value2': 'float64[pyarrow]'
    })
    result = (df_pa[df_pa['value1'] > 0]
              .groupby('category')
              .agg({'value2': ['mean', 'std'], 'value1': 'count'})
              .round(4))
    result.columns = ['avg_value', 'std_value', 'count']
    return result.sort_values('avg_value', ascending=False)


def polars_filter_aggregate(df):
    pl_df = pl.from_pandas(df)
    result = (pl_df
              .filter(pl.col('value1') > 0)
              .group_by('category')
              .agg([
        pl.mean('value2').alias('avg_value'),
        pl.std('value2').alias('std_value'),
        pl.len().alias('count')
    ])
              .sort('avg_value', descending=True))
    return result


# Операция 2: Группировка с вычислением статистик
def pandas_groupby_stats(df):
    result = df.groupby('category').agg({
        'value1': ['sum', 'mean', 'count'],
        'value2': ['sum', 'mean', 'count'],
        'value3': ['sum', 'mean', 'count']
    }).round(4)
    return result


def pandas_pyarrow_groupby_stats(df):
    df_pa = df.astype({
        'category': 'string[pyarrow]',
        'value1': 'float64[pyarrow]',
        'value2': 'float64[pyarrow]',
        'value3': 'int64[pyarrow]'
    })
    result = df_pa.groupby('category').agg({
        'value1': ['sum', 'mean', 'count'],
        'value2': ['sum', 'mean', 'count'],
        'value3': ['sum', 'mean', 'count']
    }).round(4)
    return result


def polars_groupby_stats(df):
    pl_df = pl.from_pandas(df)
    result = (pl_df
    .group_by('category')
    .agg([
        pl.sum('value1').alias('value1_sum'),
        pl.mean('value1').alias('value1_mean'),
        pl.len().alias('value1_count'),
        pl.sum('value2').alias('value2_sum'),
        pl.mean('value2').alias('value2_mean'),
        pl.len().alias('value2_count'),
        pl.sum('value3').alias('value3_sum'),
        pl.mean('value3').alias('value3_mean'),
        pl.len().alias('value3_count')
    ]))
    return result


# Операция 3: JOIN нескольких таблиц
def create_join_tables(df):
    df1 = df[['id', 'category', 'value1']].copy()
    df2 = df[['id', 'value2', 'value3']].copy()
    df3 = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'category_name': ['Type A', 'Type B', 'Type C', 'Type D'],
        'multiplier': [1.0, 1.5, 0.8, 1.2]
    })
    return df1, df2, df3


def pandas_join(df1, df2, df3):
    result = (df1.merge(df2, on='id', how='inner')
              .merge(df3, on='category', how='left'))
    return result


def pandas_pyarrow_join(df1, df2, df3):
    df1_pa = df1.astype({'category': 'string[pyarrow]'})
    df2_pa = df2.astype({})
    df3_pa = df3.astype({'category': 'string[pyarrow]'})

    result = (df1_pa.merge(df2_pa, on='id', how='inner')
              .merge(df3_pa, on='category', how='left'))
    return result


def polars_join(df1, df2, df3):
    pl_df1 = pl.from_pandas(df1)
    pl_df2 = pl.from_pandas(df2)
    pl_df3 = pl.from_pandas(df3)

    result = (pl_df1
              .join(pl_df2, on='id', how='inner')
              .join(pl_df3, on='category', how='left'))
    return result


# Операция 4: Скользящее среднее
def pandas_rolling(df):
    result = df.set_index('timestamp').sort_index()
    result['value1_rolling'] = result['value1'].rolling('10min').mean()
    result['value2_rolling'] = result['value2'].rolling('10min').mean()
    return result[['value1_rolling', 'value2_rolling']].reset_index()


def pandas_pyarrow_rolling(df):
    df_pa = df.astype({
        'value1': 'float64[pyarrow]',
        'value2': 'float64[pyarrow]'
    })
    result = df_pa.set_index('timestamp').sort_index()
    result['value1_rolling'] = result['value1'].rolling('10min').mean()
    result['value2_rolling'] = result['value2'].rolling('10min').mean()
    return result[['value1_rolling', 'value2_rolling']].reset_index()


def polars_rolling(df):
    pl_df = pl.from_pandas(df)
    result = (pl_df
              .sort('timestamp')
              .with_columns([
        pl.col('value1').rolling_mean(window_size=10).alias('value1_rolling'),
        pl.col('value2').rolling_mean(window_size=10).alias('value2_rolling')
    ])
              .select(['timestamp', 'value1_rolling', 'value2_rolling']))
    return result


# Операция 5: Resampling временных рядов
def pandas_resample(df):
    result = (df.set_index('timestamp')
              .resample('1h')
              .agg({
        'value1': 'mean',
        'value2': 'mean',
        'value3': 'mean'
    })
              .round(4)
              .reset_index())
    return result


def pandas_pyarrow_resample(df):
    df_pa = df.astype({
        'value1': 'float64[pyarrow]',
        'value2': 'float64[pyarrow]',
        'value3': 'float64[pyarrow]'
    })
    result = (df_pa.set_index('timestamp')
              .resample('1h')
              .agg({
        'value1': 'mean',
        'value2': 'mean',
        'value3': 'mean'
    })
              .round(4)
              .reset_index())
    return result


def polars_resample(df):
    pl_df = pl.from_pandas(df)
    result = (pl_df
              .sort('timestamp')
              .with_columns(pl.col('timestamp').dt.truncate('1h').alias('hour'))
              .group_by('hour')
              .agg([
        pl.mean('value1').alias('value1_mean'),
        pl.mean('value2').alias('value2_mean'),
        pl.mean('value3').alias('value3_mean')
    ])
              .rename({'hour': 'timestamp'}))
    return result


# Бенчмаркинг
def benchmark_operation(operation_func, *args):
    start_time = time.time()

    # Для измерения памяти используем более простой подход
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # в MB

    result = operation_func(*args)

    memory_after = process.memory_info().rss / 1024 / 1024  # в MB
    end_time = time.time()

    execution_time = end_time - start_time
    peak_memory = max(memory_before, memory_after)

    return execution_time, peak_memory


def run_benchmarks(df):
    operations = {
        'Фильтрация и агрегация': {
            'pandas': (pandas_filter_aggregate, [df]),
            'pandas_pyarrow': (pandas_pyarrow_filter_aggregate, [df]),
            'polars': (polars_filter_aggregate, [df])
        },
        'Группировка со статистиками': {
            'pandas': (pandas_groupby_stats, [df]),
            'pandas_pyarrow': (pandas_pyarrow_groupby_stats, [df]),
            'polars': (polars_groupby_stats, [df])
        },
        'JOIN таблиц': {
            'pandas': (pandas_join, create_join_tables(df)),
            'pandas_pyarrow': (pandas_pyarrow_join, create_join_tables(df)),
            'polars': (polars_join, create_join_tables(df))
        },
        'Скользящее среднее': {
            'pandas': (pandas_rolling, [df]),
            'pandas_pyarrow': (pandas_pyarrow_rolling, [df]),
            'polars': (polars_rolling, [df])
        },
        'Resampling': {
            'pandas': (pandas_resample, [df]),
            'pandas_pyarrow': (pandas_pyarrow_resample, [df]),
            'polars': (polars_resample, [df])
        }
    }

    results = {}
    for op_name, implementations in operations.items():
        results[op_name] = {}
        for impl_name, (func, args) in implementations.items():
            print(f"Выполняется {op_name} - {impl_name}...")
            try:
                time_taken, memory_used = benchmark_operation(func, *args)
                results[op_name][impl_name] = {
                    'time': time_taken,
                    'memory': memory_used
                }
                print(f"  Время: {time_taken:.3f}с, Память: {memory_used:.1f}MB")
            except Exception as e:
                print(f"  Ошибка: {e}")
                results[op_name][impl_name] = {
                    'time': float('inf'),
                    'memory': float('inf')
                }

    return results


def generate_comparison_report(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Подготовка данных для heatmap
    performance_data = pd.DataFrame({
        op_name: {
            impl: results[op_name][impl]['time']
            for impl in results[op_name]
        }
        for op_name in results
    }).T

    # Heatmap относительной производительности
    relative_performance = performance_data.div(performance_data.min(axis=1), axis=0)
    # Заменяем бесконечные значения на максимальные + 1
    max_val = relative_performance[relative_performance < float('inf')].max().max()
    relative_performance = relative_performance.replace(float('inf'), max_val + 1)

    sns.heatmap(relative_performance, annot=True, fmt='.2f', cmap='RdYlGn',
                center=1, ax=axes[0, 0], cbar_kws={'label': 'Относительное время выполнения'})
    axes[0, 0].set_title('Относительная производительность\n(1.0 = наилучший результат)', fontsize=12, pad=20)
    axes[0, 0].set_xlabel('Библиотека')
    axes[0, 0].set_ylabel('Операция')

    # График использования памяти
    memory_data = pd.DataFrame({
        op_name: {
            impl: results[op_name][impl]['memory']
            for impl in results[op_name]
        }
        for op_name in results
    }).T

    memory_data.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Пиковое использование памяти', fontsize=12, pad=20)
    axes[0, 1].set_xlabel('Операция')
    axes[0, 1].set_ylabel('Память (MB)')
    axes[0, 1].legend(title='Библиотека')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # График времени выполнения
    performance_data_clean = performance_data.replace(float('inf'),
                                                      performance_data[performance_data < float('inf')].max().max())
    performance_data_clean.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Время выполнения операций', fontsize=12, pad=20)
    axes[1, 0].set_xlabel('Операция')
    axes[1, 0].set_ylabel('Время (секунды)')
    axes[1, 0].legend(title='Библиотека')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Сводная таблица результатов
    summary_data = []
    for op_name in results:
        for impl_name in results[op_name]:
            time_val = results[op_name][impl_name]['time']
            memory_val = results[op_name][impl_name]['memory']
            summary_data.append({
                'Операция': op_name,
                'Библиотека': impl_name,
                'Время (с)': f"{time_val:.3f}" if time_val != float('inf') else 'Ошибка',
                'Память (MB)': f"{memory_val:.1f}" if memory_val != float('inf') else 'Ошибка'
            })

    summary_df = pd.DataFrame(summary_data)
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_df.values,
                             colLabels=summary_df.columns,
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Сводная таблица результатов', fontsize=12, pad=20)

    plt.tight_layout()
    return fig


def main():
    print("Генерация датасета...")
    df = generate_large_dataset(100000)

    print("Проверка данных:")
    print(f"Размер датасета: {df.shape}")
    print(f"Типы данных:\n{df.dtypes}")
    print(f"Пропущенные значения:\n{df.isnull().sum()}")

    print("\nЗапуск бенчмарков...")
    results = run_benchmarks(df)

    print("\nГенерация отчета...")
    fig = generate_comparison_report(results)
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Анализ удобства API
    print("\n=== Анализ удобства API ===")
    print("Pandas: Зрелый API, отличная документация, привычный синтаксис")
    print("Pandas + PyArrow: Те же преимущества pandas с улучшенной обработкой типов")
    print("Polars: Современный API, ленивые вычисления, высокая производительность")


if __name__ == "__main__":
    main()