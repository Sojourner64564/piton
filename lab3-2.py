import math
import random
from typing import Generator, List
from functools import partial
import asyncio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_points(n: int, bounds: tuple = (0, 100)) -> Generator[tuple, None, None]:
    """Генератор случайных точек в трехмерном евклидовом пространстве"""
    for _ in range(n):
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[0], bounds[1])
        z = random.uniform(bounds[0], bounds[1])
        yield (x, y, z)


def generate_roads(points: List[tuple], bidirectional_ratio: float = 0.7) -> Generator[tuple, None, None]:
    """Генератор дорог между точками с евклидовыми расстояниями"""
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Вычисляем евклидово расстояние
                dist = math.sqrt(
                    (points[i][0] - points[j][0]) ** 2 +
                    (points[i][1] - points[j][1]) ** 2 +
                    (points[i][2] - points[j][2]) ** 2
                )

                if random.random() < bidirectional_ratio:
                    # Двунаправленная связь
                    yield (i, j, dist)
                    yield (j, i, dist)
                else:
                    # Однонаправленная связь
                    yield (i, j, dist)


def calculate_distance(point1: tuple, point2: tuple) -> float:
    """Вычисление евклидова расстояния между двумя точками"""
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2
    )


def initialize_pheromones(points: List[tuple]) -> np.ndarray:
    """Инициализация матрицы феромонов"""
    n = len(points)
    return np.ones((n, n))


def calculate_transition_probability(
        pheromone: np.ndarray,
        current_point: int,
        unvisited: List[int],
        alpha: float = 1.0,
        beta: float = 2.0,
        points: List[tuple] = None
) -> List[float]:
    """Расчет вероятности перехода между точками"""
    probabilities = []
    total = 0.0

    for next_point in unvisited:
        tau = pheromone[current_point, next_point] ** alpha
        eta = (1.0 / calculate_distance(points[current_point], points[next_point])) ** beta
        prob = tau * eta
        probabilities.append(prob)
        total += prob

    if total > 0:
        probabilities = [p / total for p in probabilities]
    else:
        probabilities = [1.0 / len(unvisited)] * len(unvisited)

    return probabilities


def construct_path(pheromone: np.ndarray, points: List[tuple], alpha: float = 1.0, beta: float = 2.0) -> List[int]:
    """Построение пути муравья"""
    n = len(points)
    start_point = random.randint(0, n - 1)
    path = [start_point]
    unvisited = list(set(range(n)) - {start_point})

    while unvisited:
        current_point = path[-1]
        probabilities = calculate_transition_probability(
            pheromone, current_point, unvisited, alpha, beta, points
        )

        next_point = random.choices(unvisited, weights=probabilities)[0]
        path.append(next_point)
        unvisited.remove(next_point)

    return path


def calculate_path_length(path: List[int], points: List[tuple]) -> float:
    """Вычисление длины пути"""
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += calculate_distance(points[path[i]], points[path[i + 1]])
    # Замыкаем цикл
    total_length += calculate_distance(points[path[-1]], points[path[0]])
    return total_length


def update_pheromones(pheromone: np.ndarray, paths: List[List[int]], points: List[tuple],
                      evaporation_rate: float = 0.5, Q: float = 100.0) -> np.ndarray:
    """Обновление феромонов с испарением"""
    n = len(points)
    # Испарение
    pheromone *= (1.0 - evaporation_rate)

    # Добавление нового феромона
    for path in paths:
        path_length = calculate_path_length(path, points)
        delta_pheromone = Q / path_length

        for i in range(len(path) - 1):
            pheromone[path[i], path[i + 1]] += delta_pheromone
            pheromone[path[i + 1], path[i]] += delta_pheromone

        # Замыкаем цикл
        pheromone[path[-1], path[0]] += delta_pheromone
        pheromone[path[0], path[-1]] += delta_pheromone

    return pheromone


def ant_colony_optimization(points, num_ants: int = None, iterations: int = 100):
    """Основная логика муравьиного алгоритма"""
    if num_ants is None:
        num_ants = len(points)

    pheromone = initialize_pheromones(points)

    for iteration in range(iterations):
        # Генератор путей для каждого муравья с использованием map
        construct_path_partial = partial(construct_path, pheromone, points)
        paths = list(map(construct_path_partial, range(num_ants)))

        # Поиск лучшего пути с использованием min и partial
        calculate_length_partial = partial(calculate_path_length, points=points)
        best_path = min(paths, key=calculate_length_partial)
        best_length = calculate_length_partial(best_path)

        # Обновление феромонов
        update_pheromones_partial = partial(update_pheromones, points=points)
        pheromone = update_pheromones_partial(pheromone, paths)

        yield best_path, best_length


async def visualize_optimization(algorithm_generator, points: List[tuple]):
    """Асинхронная корутина для визуализации процесса работы алгоритма"""
    fig = plt.figure(figsize=(15, 5))

    # Настройка графиков
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')

    plt.ion()
    best_lengths = []
    iterations = []

    # Используем обычный for вместо async for для обычного генератора
    for path, length in algorithm_generator:
        # Очистка графиков
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # График 1: Трехмерный граф точек
        points_array = np.array(points)
        ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2],
                    c='blue', s=50, alpha=0.6)
        ax1.set_title('3D Граф точек')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # График 2: Динамика изменения длины маршрута
        best_lengths.append(length)
        iterations.append(len(best_lengths))
        ax2.plot(iterations, best_lengths, 'b-', linewidth=2)
        ax2.set_title('Динамика длины маршрута')
        ax2.set_xlabel('Итерация')
        ax2.set_ylabel('Длина маршрута')
        ax2.grid(True)

        # График 3: Текущий лучший маршрут
        path_points = np.array([points[i] for i in path])
        path_points = np.vstack([path_points, path_points[0]])  # Замыкаем цикл

        ax3.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2],
                    c='blue', s=50, alpha=0.6)
        ax3.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2],
                 'r-', linewidth=2, marker='o', markersize=4)
        ax3.set_title(f'Лучший маршрут (длина: {length:.2f})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

        await asyncio.sleep(0.1)  # Контроль скорости анимации

    plt.ioff()
    plt.show()


async def main():
    """Основная функция"""
    # Генерация наборов данных
    dataset_sizes = [200, 500, 1000]

    for size in dataset_sizes[:1]:  # Тестируем только с 200 точками для скорости
        print(f"Обработка набора данных из {size} точек...")

        # Генерация точек и дорог
        points = list(generate_points(size))
        roads = list(generate_roads(points, bidirectional_ratio=0.7))

        # Запуск муравьиного алгоритма
        algorithm = ant_colony_optimization(points, num_ants=20, iterations=10)

        # Визуализация
        await visualize_optimization(algorithm, points)


if __name__ == "__main__":
    asyncio.run(main())