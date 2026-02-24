import json # для работы с данными в формате json
import matplotlib.pyplot as plt # для визуализации (графики)
import random # генерация примера данных
from pathlib import Path # сохранение и импорт данных


class CreateData(): # получение / создание данных
    def example(obj, length):
        data = []
        for i in range(length):
            data.append((i, 0))
        return data

    def add_anomalies(obj, noise_probability=0.05, noise_magnitude=200):
        data = obj
        for i in range(len(obj)):
            if random.random() < noise_probability:
                el = obj[i]
                data[i] = ((el[0], el[1] + random.randint(-noise_magnitude, noise_magnitude)))
        return data

class SaveData(): # сохранение данных в файл
   def save_JSON(obj, name: str):
       path = f"{Path(__file__).parent.absolute()}/{name}.json"
       with open(path, "w", encoding="utf-8") as f:
           json.dump(obj, f, indent=4, ensure_ascii=False)

    # csv

class CollectData(): # импорт / сбор данных   
   def import_JSON(obj, file_name):
       with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
   #csv

class ShowData: # показ данных
    def show_plot(data, color="b", block=True):
        for i in range(len(data)):
            el = data[i]
            plt.plot(el[0], el[1])
            if i > 0:
                prev = data[i-1]
                plt.plot([prev[0], el[0]], [prev[1], el[1]], f'{color}-')
        plt.grid(True)
        plt.show(block=block)

    def show_hist(data, block=True): # после сглаживания серединные значения размываются и может стать сложнее опредилить аномалию
        fig, ax = plt.subplots()
        ax.hist(data, bins=8, linewidth=0.5, edgecolor="white")
        plt.show(block=block)

class ClearData: # опциональный класс для сглаживания шума
    def value_filter(val, min_val, max_val):
        return min_val <= val <= max_val
#    def percent_filter(val, percent)
    
    def average_value_neighbours(data, min_amp=0, max_amp=0): # среднее между соседними значениями # переделать амплитуду в проценты? # добавить итерации?
        orig_arr = data
        new_arr = []

        for i in range(len(data)):
            new_v = orig_arr[i][1]
            if abs(min_amp) <= abs(orig_arr[i][1]) <= abs(max_amp):
                if i == 0:
                    new_v = (orig_arr[i][1] + orig_arr[i+1][1]) / 2
                elif i == len(data) - 1:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1]) / 2
                else:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1] + orig_arr[i+1][1]) / 3
            new_arr.append((orig_arr[i][0], new_v))

        return new_arr

class Analyze: # выявление аномалий / отклонений
    def median_deviation(data): # возвращать только сильно отклоняющиеся данные?
        d = data.copy()
        for i in range(len(d)): # ЭТА ФУНКЦИЯ ПРОСТО СОРТИРУЕТ СПИСОК ПО Y!!!
            d[i] = d[i][1] # НИЧЕГО ПООРИГИНАЛЬНЕЕ НЕ ПРИДУМАЛ?
        d.sort()
        return d

    def function_derivative(data):
        return [(data[i][0], (data[i+1][1] - data[i][1]) / (data[i+1][0] - data[i][0])) for i in range(len(data) - 1)]
        for i in range(len(data)): # неверная логика
            if i == len(data) - 1: pass
            else: 
                data[i] = (data[i][0], (data[i+1][1] - data[i][1]))
        return data[:-1]



    # после сглаживания находить места отклонений и затем анализировать сырые данные в этом месте? 
    # медианное значение? # передавать медианное значение (значения в пределах нормы / те, что чаще всего) для фильтрации?


def main():
    # create
    d = []
    d = CreateData.example(d, 150)
    d = CreateData.add_anomalies(d, 1, 10) # мелкий шум
    d = CreateData.add_anomalies(d, 0.2, 30) # большой шум
    d = CreateData.add_anomalies(d, noise_magnitude=100) # аномалии

    arr = d

    dCleared = arr.copy()

    for i in range(3):
        dCleared = ClearData.average_value_neighbours(dCleared, 0, 30)
    dDerivative = Analyze.function_derivative(dCleared)

    # show
    ShowData.show_plot(arr, block=False)
    ShowData.show_plot(dCleared, color='r', block=False)
    ShowData.show_plot(dDerivative, color='g', block=True)

main()


# остальное по мере надобности


class GeneratePredict: pass # создание предположения

class CheckPredict: pass # проверка предположения

# class CheckConsistency: pass # часть CheckPredict?

class LoopManager: pass
   # цикл
