import json # для работы с данными в формате json
import matplotlib.pyplot as plt # для визуализации (графики)
import random # генерация примера данных
from dataclasses import dataclass # архитектурное решение, разделение данных и логики
from pathlib import Path # сохранение и импорт данных
import copy # скопировать график


@dataclass
class Data:
    data = None
  
class CreateData(Data): # создание данных
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
    def show_plot(data: Data, isOne=True, color="b"):
        minx = 0
        maxx = 0
        miny = 0
        maxy = 0

        for i in range(len(data.data["arr"])):
            el = data.data["arr"][i]
            if el[0] < minx: minx = el[0]
            if el[0] > maxx: maxx = el[0]

            if el[1] < miny: miny = el[1]
            if el[1] > maxy: maxy = el[1]

            plt.plot(el[0], el[1])
            if i > 0:
                prev = data.data["arr"][i-1]
                plt.plot([prev[0], el[0]], [prev[1], el[1]], f'{color}-')
        
        if isOne:
            plt.xlim(minx, maxx)
            plt.ylim(miny, maxy)
            plt.grid(True)
            plt.show()
        else:
            return minx, maxx, miny, maxy       

class ClearData: # опциональный класс для сглаживания шума
    # добавить фильтр по амплитуде шума? тип сглаживать только меньше n
#    def value_filter(min_val, max_val):
    
    def average_value_neighbours(obj: Data, max_amp=0): # среднее между соседними значениями
        orig_arr = obj.data["arr"]
        new_arr = []

        for i in range(len(obj.data["arr"])):
            new_v = orig_arr[i][1]
            if abs(orig_arr[i][1]) <= max_amp:
                if i == 0:
                    new_v = (orig_arr[i][1] + orig_arr[i+1][1]) / 2
                elif i == len(obj.data["arr"]) - 1:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1]) / 2
                else:
                    new_v = (orig_arr[i-1][1] + orig_arr[i][1] + orig_arr[i+1][1]) / 3
            new_arr.append((orig_arr[i][0], new_v))

        obj.data["arr"] = new_arr
    
    # медианное значение?

d = Data()
CreateData.example(d, 100, 10)
CreateData.add_anomalies(d)

dCleared = copy.deepcopy(d)

for i in range(5):
    ClearData.average_value_neighbours(dCleared, 10)

ShowData.show_plot(d)
ShowData.show_plot(dCleared)




# остальное по мере надобности



class Analyze: pass # выявление аномалий / отклонений
    # после сглаживания находить места отклонений и затем анализировать сырые данные в этом месте? 

class GeneratePredict: pass # создание предположения

class CheckPredict: pass # проверка предположения

# class CheckConsistency: pass

class LoopManager: pass
   # цикл
