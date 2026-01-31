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

d = Data()
CreateData.example_with_anomalies(d, 250, 10, 0.1, 50)
ShowData.ShowPlot(d.data["x"], d.data["y"])






# остальное по мере надобности

class Clear: pass # опциональный класс для сглаживания шума

class Analyze: pass # выявление аномалий / отклонений

class GeneratePredict: pass # создание предположения

class CheckPredict: pass # проверка предположения

# class CheckConsistency: pass

class LoopManager: pass
   # цикл
