class DataCollector: # сбор данных
    # заранее собранные данные

    # getSpecificData получить конкретные необходимые данные
    # (из внешних источников или своих источников(автоматизировать эксперименты?))
    pass


class Analyze: # анализ данных
    # найти отклонения? закономерности? (Exploratory Analysis)
    # выделение значимых атрибутов данных? (Feature Extraction)
    def __init__(self):
        self.data = [] # raw data
        pass
    pass


class GeneratePredict: # создание предположений на основе анализа
    # обобщить отклонения / закономерности?
    # возможные инструменты: LLM, генетические алгоритмы, символьная регрессия  ...
    def __init__(self):
        self.data = []
        # self.method = None # метод нахождения? по типу индуктивный или что то такое?
        # self.generator = None # сам алгоритм, символьная регрессия / LLM / комбинация алгоритмов / ... 
        pass
    pass

class CheckPredict: # проверка гипотезы / предсказания / теории на работоспособность и предсказательную силу
    # симуляция на данных не участвовавших в анализе (выделить проверочные данные изначально или добавлять по мере исследования)?
    # метрики оценки? простота, предиктивная способность? 
    pass

class CheckConsistency: # проверка непротиворечивости модели
    # логические / формальные анализаторы
    # проверка причинно следственной связи (часть прошлого пункта)
    # knowledge base (аксиоматика) !!! ОСТОРОЖНО использовать, изначально могут быть ложные / неполные аксиомы, которые приведут к ложным выводам (по типу нож - острый, ножу желательно быть таким, но не все и не всегда такие), к тому же некоторые аксиомы тоже стоит подвергать сомнению
    pass

class LoopManager: # основной цикл исследования, объединяет предыдущие шаги
    # цикл (сбор, анализ данных) -> генерация предсказания(гипотезы) -> проверка гипотезы
    # анализ данных больше относится к сбору или генерации? или это отдельная вещь?
    isConsistant = False
    predict = None
    data = DataCollector # или лучше внести в основной цикл? 
    while CheckConsistency != True:
        # data = data.clear # возможно стоит? или лучше анализировать сырые? очистив данные можно убрать ключевые детали, но и к сырым тоже вопросы
        peculiarities = data.analyze # нахождение отклоняющихся данных


        if GeneratePredict(peculiarities): # если данных хватает
            new_predict = GeneratePredict(peculiarities)
        else: # если данных не хватает
            while GeneratePredict(peculiarities) returns low_data: 
                data += DataCollector(requested_data)


        # что может лучше сначала проверить непротиворечивость, а затем проводить симуляцию?
        if CheckPredict(new_predict) > predict: # записать предыдущую
            file.write(new_predict, predict, "better", reason)
            predict = new_predict
        if CheckPredict(new_predict) == predict: # записать как альтернативу(равносильную)
            file.write(new_predict, predict, "equal", reason)
        if CheckPredict(new_predict) < predict: # записать неудачу, по возможности что пошло не так
            file.write(new_predict, predict, "worse", reason)
        
        isConsistant = CheckConsistency(predict)
    pass