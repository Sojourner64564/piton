import numpy as np
import time
import memory_profiler
import matplotlib.pyplot as plt
import seaborn as sns
import math

def generate_test_datasets() -> dict:
    """Генерация тестовых наборов данных"""
    return {
        '10^3': np.random.random(1000),
        '10^4': np.random.random(10000),
        '10^5': np.random.random(100000),
        '10^6': np.random.random(1000000)
    }


# Чистый Python версии
def py_square(data):
    return [x ** 2 for x in data]


def py_sin(data):
    return [math.sin(x) for x in data]


def py_sum(data):
    return sum(data)


def py_max(data):
    return max(data)


# NumPy версии
def np_square(data):
    return np.square(data)


def np_sin(data):
    return np.sin(data)


def np_sum(data):
    return np.sum(data)


def np_max(data):
    return np.max(data)


def benchmark_operations():
    """Бенчмарк операций для сравнения производительности"""
    datasets = generate_test_datasets()
    operations = {
        'square': {'py': py_square, 'np': np_square},
        'sin': {'py': py_sin, 'np': np_sin},
        'sum': {'py': py_sum, 'np': np_sum},
        'max': {'py': py_max, 'np': np_max}
    }

    results = {}

    for op_name, op_funcs in operations.items():
        results[op_name] = {}
        for impl_name, op_func in op_funcs.items():
            results[op_name][impl_name] = {}

            for dataset_name, data in datasets.items():
                start_time = time.time()
                op_func(data)
                end_time = time.time()

                mem_usage = memory_profiler.memory_usage((op_func, (data,)))

                results[op_name][impl_name][dataset_name] = {
                    'time': end_time - start_time,
                    'memory': max(mem_usage) - min(mem_usage)
                }

    return results


def plot_performance(results):
    """Построение графиков производительности"""

    # График времени выполнения
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    dataset_sizes = ['10^3', '10^4', '10^5', '10^6']
    operations = list(results.keys())

    for i, op_name in enumerate(operations):
        py_times = [results[op_name]['py'][size]['time'] for size in dataset_sizes]
        np_times = [results[op_name]['np'][size]['time'] for size in dataset_sizes]

        # o- кружки с линиями для Python
        # s- квадраты с линиями для NumPy
        axes[i].plot(dataset_sizes, py_times, 'o-', label='Pure Python', linewidth=2)
        axes[i].plot(dataset_sizes, np_times, 's-', label='NumPy', linewidth=2)
        axes[i].set_title(f'Время выполнения: {op_name}')
        axes[i].set_xlabel('Размер данных')
        axes[i].set_ylabel('Время (секунды)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # График потребления памяти
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, op_name in enumerate(operations):
        py_memory = [results[op_name]['py'][size]['memory'] for size in dataset_sizes]
        np_memory = [results[op_name]['np'][size]['memory'] for size in dataset_sizes]

        axes[i].plot(dataset_sizes, py_memory, 'o-', label='Pure Python', linewidth=2)
        axes[i].plot(dataset_sizes, np_memory, 's-', label='NumPy', linewidth=2)
        axes[i].set_title(f'Потребление памяти: {op_name}')
        axes[i].set_xlabel('Размер данных')
        axes[i].set_ylabel('Память (MiB)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Тепловая карта
    speedup_matrix = []
    for op_name in operations:
        speedup_row = []
        for size in dataset_sizes:
            py_time = results[op_name]['py'][size]['time']
            np_time = results[op_name]['np'][size]['time']
            # чтобы не было дления на ноль
            if np_time > 0:
                speedup = py_time / np_time
            else:
                speedup = 1.0
            speedup_row.append(speedup)
        speedup_matrix.append(speedup_row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(speedup_matrix,
                annot=True,
                fmt='.1f',
                xticklabels=dataset_sizes,
                yticklabels=[f'{op}' for op in operations],
                cmap='YlOrRd',
                cbar_kws={'label': 'Коэффициент ускорения (Python/NumPy)'})
    plt.title('Тепловая карта относительного ускорения NumPy над чистым Python')
    plt.xlabel('Размер данных')
    plt.ylabel('Операция')
    plt.show()



if __name__ == "__main__":
    print("Запуск бенчмарка...")
    results = benchmark_operations()

    plot_performance(results)