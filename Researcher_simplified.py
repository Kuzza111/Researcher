import json # для работы с данными в формате json
import matplotlib.pyplot as plt # для визуализации (графики)
import random # генерация примера данных
from dataclasses import dataclass # архитектурное решение, разделение данных и логики
from pathlib import Path # сохранение и импорт данных
import copy # скопировать график
import numpy as np


@dataclass
class Data:
    data = None
  
class CreateData(Data): # получение / создание данных
    def example(obj: Data, length, ran):
        data = []

        for i in range(length):
            data.append((i, random.randint(-ran, ran)))

        obj.data = {"arr": data}

    def add_anomalies(obj, noise_probability=0.05, noise_magnitude=200):
        for i in range(len(obj.data["arr"])):
            if random.random() < noise_probability:
                el = obj.data["arr"][i]
                obj.data["arr"][i] = (el[0], el[1] + random.randint(-noise_magnitude, noise_magnitude))

class SaveData(Data): # сохранение данных в файл
   def save_JSON(obj: Data, name: str):
       path = f"{Path(__file__).parent.absolute()}/{name}.json"
       with open(path, "w", encoding="utf-8") as f:
           json.dump(obj.data, f, indent=4, ensure_ascii=False)

    # csv

class CollectData(Data): # импорт / сбор данных   
   def import_JSON(obj: Data, file_name):
       with open(file_name, "r", encoding="utf-8") as f:
           obj.data = json.load(f)
   #csv

class ShowData: # показ данных
    def show_plot(data: Data, color="b", block=True):
        for i in range(len(data.data["arr"])):
            el = data.data["arr"][i]
            plt.plot(el[0], el[1])
            if i > 0:
                prev = data.data["arr"][i-1]
                plt.plot([prev[0], el[0]], [prev[1], el[1]], f'{color}-')
        plt.grid(True)
        plt.show(block=block)

    def show_hist(obj: Data, block=True):
        fig, ax = plt.subplots()
        ax.hist(obj.data, bins=8, linewidth=0.5, edgecolor="white")
        plt.show(block=block)

class ClearData: # опциональный класс для сглаживания шума
#    def value_filter(val, min_val, max_val): min_val <= val <= max_val
#    def percent_filter(val, percent)
    
    def average_value_neighbours(obj: Data, min_amp=0, max_amp=0): # среднее между соседними значениями # переделать амплитуду в проценты?
        orig_arr = obj.data["arr"]
        new_arr = []

        for i in range(len(obj.data["arr"])):
            new_v = orig_arr[i][1]
            if abs(min_amp) <= abs(orig_arr[i][1]) <= abs(max_amp):
                if i == 0:
                    new_v = (orig_arr[i][1] + orig_arr[i+1][1]) / 2
                elif i == len(obj.data["arr"]) - 1:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1]) / 2
                else:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1] + orig_arr[i+1][1]) / 3
            new_arr.append((orig_arr[i][0], new_v))

        obj.data["arr"] = new_arr

class Analyze: # выявление аномалий / отклонений
    def median_deviation(obj): # возвращать только сильно отклоняющиеся данные?
        data = copy.deepcopy(obj.data["arr"])
        for i in range(len(data)):
            data[i] = data[i][1]
        data.sort()
        obj.data = data

        


    # после сглаживания находить места отклонений и затем анализировать сырые данные в этом месте? 
    # отслеживать скорость изменения функции? (насколько отличаются соседние значения?)
    # медианное значение? # передавать медианное значение (значения в пределах нормы / те, что чаще всего) для фильтрации?


def main():
    d = Data()
    CreateData.example(d, 100, random.randint(0, 25))
    CreateData.add_anomalies(d) # большие аномалии
    CreateData.add_anomalies(d, 0.25, 50) # большой шум

    dCleared = copy.deepcopy(d)
    #dHist = copy.deepcopy(d)
    #dHistCleared = copy.deepcopy(d)


    for i in range(3):
        ClearData.average_value_neighbours(dCleared, 0, 25)
    #    ClearData.average_value_neighbours(dHistCleared, 0, 10)
    #Analyze.median_deviation(dHist)
    #Analyze.median_deviation(dHistCleared)

    ShowData.show_plot(d, block=False)
    ShowData.show_plot(dCleared, color='r', block=True)
    #ShowData.show_hist(dHist)
    #ShowData.show_hist(dHistCleared) # после сглаживания серединные значения размываются и может стать сложнее опредилить аномалию

main()


# остальное по мере надобности


class GeneratePredict: pass # создание предположения

class CheckPredict: pass # проверка предположения

# class CheckConsistency: pass # часть CheckPredict?

class LoopManager: pass
   # цикл
