import json
from functools import reduce


def load_countries(filename):
    """Загрузка данных о странах из JSON файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return []
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON в файле {filename}")
        return []


def task_1_2_map_uppercase(countries):
    """Преобразование названий стран к верхнему регистру с помощью map()"""
    return list(map(lambda country: country.upper(), countries))


def task_1_3_filter_countries(countries):
    """Фильтрация стран по различным условиям с помощью filter()"""
    # Страны содержащие подстроку 'land'
    land_countries = list(filter(lambda country: 'land' in country.lower(), countries))

    # Страны с ровно 6 символами в названии
    six_chars_countries = list(filter(lambda country: len(country) == 6, countries))

    # Страны с 6 и более буквами
    six_or_more_letters = list(filter(lambda country: len(country) >= 6, countries))

    # Страны начинающиеся с буквы 'E'
    starts_with_e = list(filter(lambda country: country.lower().startswith('e'), countries))

    return {
        'land_countries': land_countries,
        'six_chars_countries': six_chars_countries,
        'six_or_more_letters': six_or_more_letters,
        'starts_with_e': starts_with_e
    }


def task_1_4_reduce_nordic(countries):
    """Объединение стран Северной Европы с помощью reduce()"""
    nordic_countries = ['Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
    nordic_in_list = list(filter(lambda country: country in nordic_countries, countries))

    if len(nordic_in_list) > 1:
        result = reduce(lambda x, y: f"{x}, {y}",
                        nordic_in_list[:-1]) + f" и {nordic_in_list[-1]} являются странами Северной Европы"
        return result
    elif len(nordic_in_list) == 1:
        return f"{nordic_in_list[0]} является страной Северной Европы"
    else:
        return "Страны Северной Европы не найдены"


def task_1_5_without_hof(countries):
    """Реализация тех же задач без функций высшего порядка"""
    # Преобразование к верхнему регистру
    uppercase_countries = [country.upper() for country in countries]

    # Фильтрация
    land_countries = [country for country in countries if 'land' in country.lower()]
    six_chars_countries = [country for country in countries if len(country) == 6]
    six_or_more_letters = [country for country in countries if len(country) >= 6]
    starts_with_e = [country for country in countries if country.lower().startswith('e')]

    # Объединение стран Северной Европы
    nordic_countries = ['Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
    nordic_in_list = [country for country in countries if country in nordic_countries]

    nordic_string = ""
    if len(nordic_in_list) > 1:
        nordic_string = ", ".join(nordic_in_list[:-1]) + f" и {nordic_in_list[-1]} являются странами Северной Европы"
    elif len(nordic_in_list) == 1:
        nordic_string = f"{nordic_in_list[0]} является страной Северной Европы"
    else:
        nordic_string = "Страны Северной Европы не найдены"

    return {
        'uppercase_countries': uppercase_countries,
        'land_countries': land_countries,
        'six_chars_countries': six_chars_countries,
        'six_or_more_letters': six_or_more_letters,
        'starts_with_e': starts_with_e,
        'nordic_string': nordic_string
    }


def task_1_6_categorize_curried():
    """Каррирование через lambda для категоризации стран"""
    categorize_curried = lambda pattern: lambda countries: [c for c in countries if pattern in c.lower()]
    return categorize_curried


def task_1_7_categorize_closure():
    """То же самое через замыкания"""

    def categorize_closure(pattern):
        def categorize(countries):
            return [country for country in countries if pattern in country.lower()]

        return categorize

    return categorize_closure


def task_1_8_countries_data_analysis():
    """Анализ данных из countries-data.json"""
    try:
        with open('countries-data.json', 'r', encoding='utf-8') as file:
            countries_data = json.load(file)
    except FileNotFoundError:
        print("Файл countries-data.json не найден")
        return
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON в файле countries-data.json")
        return

    # Сортировка стран по названию
    sorted_by_name = sorted(countries_data, key=lambda country: country.get('name', ''))

    # Сортировка стран по столице
    sorted_by_capital = sorted(countries_data, key=lambda country: country.get('capital', ''))

    # Сортировка стран по населению (по убыванию)
    sorted_by_population = sorted(countries_data, key=lambda country: country.get('population', 0), reverse=True)

    # Поиск 10 самых распространенных языков
    language_count = {}
    for country in countries_data:
        languages = country.get('languages', [])
        for language in languages:
            language_count[language] = language_count.get(language, 0) + 1

    top_10_languages = sorted(language_count.items(), key=lambda x: x[1], reverse=True)[:10]

    # Страны, где говорят на самых распространенных языках
    languages_countries = {}
    for language, _ in top_10_languages:
        countries_speaking = [country['name'] for country in countries_data if language in country.get('languages', [])]
        languages_countries[language] = countries_speaking

    # 10 самых населенных стран
    top_10_populated = sorted_by_population[:10]

    return {
        'sorted_by_name': sorted_by_name,
        'sorted_by_capital': sorted_by_capital,
        'sorted_by_population': sorted_by_population,
        'top_10_languages': top_10_languages,
        'languages_countries': languages_countries,
        'top_10_populated': top_10_populated
    }


def main():
    """Основная функция выполнения задания"""
    # Загрузка данных
    countries = load_countries(r'countries.json')

    if not countries:
        print("Не удалось загрузить данные о странах")
        return

    print("=== Задание 1.2: Преобразование к верхнему регистру ===")
    uppercase_countries = task_1_2_map_uppercase(countries)
    print(f"Первые 5 стран: {uppercase_countries[:5]}")

    print("\n=== Задание 1.3: Фильтрация стран ===")
    filtered_results = task_1_3_filter_countries(countries)
    print(f"Страны с 'land': {filtered_results['land_countries'][:5]}")
    print(f"Страны с 6 символами: {filtered_results['six_chars_countries'][:5]}")
    print(f"Страны с 6+ буквами: {len(filtered_results['six_or_more_letters'])} стран")
    print(f"Страны на 'E': {filtered_results['starts_with_e'][:5]}")

    print("\n=== Задание 1.4: Объединение стран Северной Европы ===")
    nordic_result = task_1_4_reduce_nordic(countries)
    print(nordic_result)

    print("\n=== Задание 1.5: Без функций высшего порядка ===")
    without_hof_results = task_1_5_without_hof(countries)
    print(f"Страны с 'land' (без HOF): {without_hof_results['land_countries'][:5]}")

    print("\n=== Задание 1.6: Каррирование ===")
    categorize_curried = task_1_6_categorize_curried()
    land_countries_curried = categorize_curried('land')(countries)
    ia_countries_curried = categorize_curried('ia')(countries)
    print(f"Страны с 'land' (каррирование): {land_countries_curried[:5]}")
    print(f"Страны с 'ia' (каррирование): {ia_countries_curried[:5]}")

    print("\n=== Задание 1.7: Замыкания ===")
    categorize_closure = task_1_7_categorize_closure()
    land_countries_closure = categorize_closure('land')(countries)
    island_countries_closure = categorize_closure('island')(countries)
    print(f"Страны с 'land' (замыкания): {land_countries_closure[:5]}")
    print(f"Страны с 'island' (замыкания): {island_countries_closure[:5]}")

    print("\n=== Задание 1.8: Анализ данных countries-data.json ===")
    analysis_results = task_1_8_countries_data_analysis()

    if analysis_results:
        print("10 самых населенных стран:")
        for i, country in enumerate(analysis_results['top_10_populated'], 1):
            print(f"{i}. {country.get('name', 'N/A')}: {country.get('population', 0):,}")

        print("\n10 самых распространенных языков:")
        for i, (language, count) in enumerate(analysis_results['top_10_languages'], 1):
            print(f"{i}. {language}: {count} стран")


if __name__ == "__main__":
    main()
