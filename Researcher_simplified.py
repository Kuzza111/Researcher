import json
import matplotlib.pyplot as plt
import numpy as np
import random

class Data:
    def __init__(self):
        self.data = None

class CreateData(Data):
    def example(ran): # self
        x = np.arange(ran)
        y = np.array([random.randint(0, 5) for _ in range(ran)])
        data = [x, y]
        return data # заменить на установку self.data

class CollectData(Data): # импорт / сбор данных    
    def import_JSON(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    #csv

class SaveData(Data):
    def save_JSON(self, name):
        with open(f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
    #csv

class ShowData:
    def __init__(self):
        self.data = None
    
    def ShowPlot(x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

#d = Data()
data = CreateData.example(100)
ShowData.ShowPlot(data[0], data[1])










# остальное по мере надобности

class Clear: pass # опциональный класс для сглаживания шума

class Analyze: pass # выявление аномалий / отклонений

class GeneratePredict: pass
    # создание предположения

class CheckPredict: pass
    # проверка предположения

# class CheckConsistency: pass

class LoopManager: pass
    # цикл