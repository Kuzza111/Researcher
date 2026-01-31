import json # для работы с данными в формате json
import matplotlib.pyplot as plt # для визуализации (графики)
import random # генерация примера данных
from dataclasses import dataclass # архитектурное решение, разделение данных и логики
from pathlib import Path # сохранение и импорт данных


@dataclass
class Data:
    data = None
  
class CreateData(Data):
    def example(obj: Data, length, ran):
        x = []
        for i in range(length):
            x.append(i)
        y = []
        for i in range(length):
            y.append(random.randint(-ran, ran))
        
        obj.data = {"x": x, "y": y}

    def example_with_anomalies(obj, length, ran, noise_probability=0.05, noise_magnitude=200):
        CreateData.example(obj, length, ran)
    
        for i in range(len(obj.data["y"])):
            if random.random() < noise_probability:  # ~5% случаев
                noise = random.randint(-noise_magnitude, noise_magnitude)
                obj.data["y"][i] += noise

    

class SaveData(Data):
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

class ShowData:
   def ShowPlot(x, y):
       fig, ax = plt.subplots()
       ax.plot(x, y)
       plt.show()

class ClearData: # опциональный класс для сглаживания шума
    # добавить фильтр по амплитуде шума? тип сглаживать только меньше n
    def average_value_neighbours(obj: Data): # среднее между соседними значениями
        original = obj.data

        for i in range(len(obj.data["y"])):
            if i == 0: 
                obj.data["y"][i] = (original["y"][i] + original["y"][i+1]) / 2
            elif i == len(obj.data["y"]) - 1:
                obj.data["y"][i] = (original["y"][i-1] + original["y"][i]) / 2
            else:
                obj.data["y"][i] = (original["y"][i-1] + original["y"][i] + original["y"][i+1]) / 3
    # медианное значение?

d = Data()
CreateData.example_with_anomalies(d, 250, 10, 0.1, 50)
ShowData.ShowPlot(d.data["x"], d.data["y"])
ClearData.average_value_neighbours(d)
ShowData.ShowPlot(d.data["x"], d.data["y"])
ClearData.average_value_neighbours(d)
ShowData.ShowPlot(d.data["x"], d.data["y"])
ClearData.average_value_neighbours(d)
ShowData.ShowPlot(d.data["x"], d.data["y"])





# остальное по мере надобности



class Analyze: pass # выявление аномалий / отклонений
    # после сглаживания находить места отклонений и затем анализировать сырые данные в этом месте? 

class GeneratePredict: pass # создание предположения

class CheckPredict: pass # проверка предположения

# class CheckConsistency: pass

class LoopManager: pass
   # цикл
