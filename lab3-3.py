import time
from functools import wraps, reduce, lru_cache
import random


# 1. Декоратор для измерения времени выполнения
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result

    return wrapper


# 2. Мемоизация тяжелых вычислений
@lru_cache(maxsize=None)
def heavy_computation(n):
    return sum(i * i for i in range(n))


# 3. Pipeline обработки данных
def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


# Функции для pipeline
def normalize_data(data):
    max_val = max(data) if data else 1
    return list(map(lambda x: x / max_val, data))


def calculate_metrics(data):
    mean = sum(data) / len(data) if data else 0
    variance = sum(map(lambda x: (x - mean) ** 2, data)) / len(data) if data else 0
    return {"mean": mean, "variance": variance}


def generate_report(metrics):
    return f"Среднее: {metrics['mean']:.4f}, Дисперсия: {metrics['variance']:.4f}"


analysis_pipeline = compose(
    generate_report,
    calculate_metrics,
    normalize_data
)


# 4. Сравнительный анализ на разных размерах данных
@timing_decorator
def analyze_dataset(data):
    return analysis_pipeline(data)


def generate_test_data(size):
    return [random.uniform(0, 100) for _ in range(size)]


def comparative_analysis():
    sizes = [200, 500, 1000]

    for size in sizes:
        print(f"\nАнализируется датасет размером {size}:")
        test_data = generate_test_data(size)
        result = analyze_dataset(test_data)
        print(result)


if __name__ == "__main__":
    comparative_analysis()