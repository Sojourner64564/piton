import numpy as np
import scipy
import scipy.optimize
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


# 1. ОПТИМИЗАЦИЯ

def rosenbrock(x):
    """Функция Розенброка"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def benchmark_optimization():
    methods = ['BFGS', 'CG', 'Nelder-Mead', 'Powell']
    results = {}

    for method in methods:
        start_time = time.time()
        result = scipy.optimize.minimize(rosenbrock,
                                         x0=np.random.random(10) * 2,
                                         method=method)
        end_time = time.time()

        results[method] = {
            'time': end_time - start_time,
            'iterations': result.nit,
            'success': result.success,
            'minimum': result.fun
        }

    return results


# Запуск оптимизации
print("=== ОПТИМИЗАЦИЯ ФУНКЦИИ РОЗЕНБРОКА ===")
optimization_results = benchmark_optimization()

# Вывод результатов
for method, res in optimization_results.items():
    print(f"\nМетод: {method}")
    print(f"Время: {res['time']:.4f} сек")
    print(f"Итерации: {res['iterations']}")
    print(f"Успех: {res['success']}")
    print(f"Минимум: {res['minimum']:.6f}")

# 2. Поиск корней системы уравнений
print("\n=== ПОИСК КОРНЕЙ СИСТЕМЫ УРАВНЕНИЙ ===")


def system_equations(x):
    """Система нелинейных уравнений"""
    return [
        x[0] ** 2 + x[1] ** 2 - 4,
        np.exp(x[0]) + x[1] - 1
    ]


# Начальное приближение
x0 = np.array([1.0, 1.0])
root_result = scipy.optimize.root(system_equations, x0, method='hybr')

print(f"Корни системы: {root_result.x}")
print(f"Успех: {root_result.success}")
print(f"Значения функций в корнях: {system_equations(root_result.x)}")

# 3. Линейное программирование
print("\n=== ЛИНЕЙНОЕ ПРОГРАММИРОВАНИЕ ===")

# Целевая функция: минимизировать c^T * x
c = np.array([-1, -2])  # Коэффициенты целевой функции

# Ограничения: A_ub * x <= b_ub
A_ub = np.array([[1, 1],  # x + y <= 4
                 [2, 1]])  # 2x + y <= 5
b_ub = np.array([4, 5])

# Границы переменных
bounds = [(0, None), (0, None)]  # x >= 0, y >= 0

# Решение задачи линейного программирования
lp_result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

print(f"Оптимальное решение: {lp_result.x}")
print(f"Значение целевой функции: {lp_result.fun}")
print(f"Успех: {lp_result.success}")

# 2. ЦИФРОВАЯ ОБРАБОТКА СИГНАЛА

print("\n=== ЦИФРОВАЯ ОБРАБОТКА СИГНАЛА ===")

# 1. Генерация тестового сигнала
fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, fs, endpoint=False)  # Временная ось

# Чистый сигнал: сумма двух синусоид
freq1 = 50  # Частота 1
freq2 = 120  # Частота 2
clean_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Добавление шума
np.random.seed(42)
noise = 0.5 * np.random.normal(size=clean_signal.shape)
noisy_signal = clean_signal + noise

# 2. Фурье-анализ для выделения частот
fft_clean = np.fft.fft(clean_signal)
fft_noisy = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(clean_signal), 1 / fs)

# 3. Применение фильтра для очистки сигнала
# Создание полосового фильтра для выделения основной частоты
nyquist = fs / 2
lowcut = 40  # Нижняя граница частоты
highcut = 130  # Верхняя граница частоты

# Создание фильтра Баттерворта
b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# 4. Визуализация результатов

# Создание фигур для графиков
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График 1: Сигналы во временной области
axes[0, 0].plot(t, clean_signal, 'b-', alpha=0.7, label='Чистый сигнал')
axes[0, 0].plot(t, noisy_signal, 'r-', alpha=0.5, label='Зашумленный сигнал')
axes[0, 0].plot(t, filtered_signal, 'g-', linewidth=2, label='Фильтрованный сигнал')
axes[0, 0].set_xlabel('Время (с)')
axes[0, 0].set_ylabel('Амплитуда')
axes[0, 0].set_title('Сигналы во временной области')
axes[0, 0].legend()
axes[0, 0].grid(True)

# График 2: Спектры сигналов
positive_freq = frequencies > 0
axes[0, 1].plot(frequencies[positive_freq], np.abs(fft_clean[positive_freq]),
                'b-', alpha=0.7, label='Спектр чистого сигнала')
axes[0, 1].plot(frequencies[positive_freq], np.abs(fft_noisy[positive_freq]),
                'r-', alpha=0.5, label='Спектр зашумленного сигнала')
axes[0, 1].set_xlabel('Частота (Гц)')
axes[0, 1].set_ylabel('Амплитуда')
axes[0, 1].set_title('Частотный спектр')
axes[0, 1].legend()
axes[0, 1].grid(True)

# График 3: Детальное сравнение чистого и фильтрованного сигнала
axes[1, 0].plot(t, clean_signal, 'b-', alpha=0.7, label='Чистый сигнал')
axes[1, 0].plot(t, filtered_signal, 'g-', linewidth=2, label='Фильтрованный сигнал')
axes[1, 0].set_xlabel('Время (с)')
axes[1, 0].set_ylabel('Амплитуда')
axes[1, 0].set_title('Сравнение чистого и фильтрованного сигнала')
axes[1, 0].legend()
axes[1, 0].grid(True)

# График 4: Ошибка фильтрации
error = clean_signal - filtered_signal
axes[1, 1].plot(t, error, 'm-', label='Ошибка фильтрации')
axes[1, 1].set_xlabel('Время (с)')
axes[1, 1].set_ylabel('Амплитуда')
axes[1, 1].set_title('Ошибка между чистым и фильтрованным сигналом')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Вывод статистики по обработке сигнала
print("\n=== СТАТИСТИКА ОБРАБОТКИ СИГНАЛА ===")
print(f"Среднеквадратичная ошибка до фильтрации: {np.sqrt(np.mean((clean_signal - noisy_signal) ** 2)):.4f}")
print(f"Среднеквадратичная ошибка после фильтрации: {np.sqrt(np.mean((clean_signal - filtered_signal) ** 2)):.4f}")
print(f"Улучшение SNR: {10 * np.log10(np.var(clean_signal) / np.var(clean_signal - filtered_signal)):.2f} dB")