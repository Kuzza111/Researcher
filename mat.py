#!/usr/bin/env python3
"""
Скрипт для последовательного дообучения модели на нескольких датасетах.
Формат данных: проблема - решение.
После каждого датасета модель сохраняется, а данные удаляются из памяти.
"""

import os
import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, DatasetDict
import logging

# ==================== КОНФИГУРАЦИЯ ====================
# 1. МОДЕЛЬ (указать название с Hugging Face)
BASE_MODEL_NAME = "google/flan-t5-small"  # Инструктированная модель

# 2. СПИСОК ДАТАСЕТОВ (указать названия с Hugging Face)
DATASET_NAMES = [
    "qwedsacf/competition_math",
    # "another/dataset",  # Добавьте другие датасеты по необходимости
]

# 3. ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ
TRAINING_CONFIG = {
    "num_train_epochs": 3,           # Количество эпох на каждый датасет
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 50,             # Частота логирования
    "eval_strategy": "epoch",        # Оценка после каждой эпохи
    "save_strategy": "no",          # Не сохранять чекпоинты (сохраним один раз вручную)
    "predict_with_generate": True,
    "fp16": torch.cuda.is_available(),  # Использовать mixed precision если доступно
}

# 4. ПАРАМЕТРЫ ДАННЫХ
MAX_INPUT_LENGTH = 512    # Максимальная длина входной последовательности
MAX_TARGET_LENGTH = 256   # Максимальная длина выходной последовательности
PROMPT_TEMPLATE = "Реши задачу: {}"  # Шаблон для создания промпта
TEST_SIZE = 0.1           # Доля данных для валидации

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ======================================================

def create_prompt(example):
    """Создание промпта из примера с проблемой и решением."""
    # Предполагаем, что датасет имеет поля 'problem' и 'solution'
    # Если названия полей отличаются, нужно изменить здесь
    problem = example.get('problem', '')
    solution = example.get('solution', '')
    
    # Если нет нужных полей, пробуем найти альтернативные названия
    if not problem or not solution:
        for key, value in example.items():
            if 'problem' in key.lower() and not problem:
                problem = value
            if 'solution' in key.lower() and not solution:
                solution = value
    
    # Форматируем промпт
    input_text = PROMPT_TEMPLATE.format(problem)
    target_text = solution
    
    return {'input_text': input_text, 'target_text': target_text}

def tokenize_function(examples, tokenizer):
    """Токенизация примеров."""
    # Токенизация входных текстов
    inputs = tokenizer(
        examples['input_text'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Токенизация целевых текстов
    # Для T5 токенизатора нет метода as_target_tokenizer
    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, 'as_target_tokenizer') else open('/dev/null') as _:
        labels = tokenizer(
            examples['target_text'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
    
    inputs['labels'] = labels['input_ids']
    return inputs

def cleanup_dataset(dataset):
    """Полная очистка датасета из памяти и с диска."""
    logger.info("Очистка датасета из памяти...")
    
    # Удаляем все ссылки на датасет
    del dataset
    gc.collect()
    
    # Очищаем кэш GPU если доступен
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Кэш GPU очищен")
    
    # Принудительный сбор мусора
    gc.collect()
    logger.info("Очистка завершена")

def train_on_dataset(model_name, dataset_name, output_dir, training_args):
    """Обучение модели на одном датасете."""
    logger.info(f"Начинаю обучение на датасете: {dataset_name}")
    
    # Загрузка токенизатора
    logger.info(f"Загрузка токенизатора из {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Загрузка датасета
    logger.info(f"Загрузка датасета {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Ошибка загрузки датасета {dataset_name}: {e}")
        return None
    
    # Проверка структуры датасета
    logger.info(f"Структура датасета: {dataset}")
    
    # Проверяем наличие нужных полей в первом примере
    if isinstance(dataset, DatasetDict) and 'train' in dataset:
        sample = dataset['train'][0]
    elif isinstance(dataset, DatasetDict) and len(dataset) > 0:
        # Берем первый доступный сплит
        first_split = list(dataset.keys())[0]
        sample = dataset[first_split][0]
    else:
        sample = dataset[0]
    
    logger.info(f"Поля в примере датасета: {list(sample.keys())}")
    
    # Определяем доступные сплиты
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        logger.info(f"Доступные сплиты: {splits}")
        # Используем 'train' если есть, иначе первый сплит
        target_split = 'train' if 'train' in splits else splits[0]
    else:
        target_split = None
    
    # Применяем функцию создания промптов
    logger.info("Создание промптов...")
    if target_split:
        dataset = dataset.map(create_prompt, batched=False)
        # Получаем список колонок для удаления
        columns_to_remove = dataset[target_split].column_names
    else:
        dataset = dataset.map(create_prompt, batched=False)
        columns_to_remove = dataset.column_names
    
    # Токенизация
    logger.info("Токенизация данных...")
    if target_split:
        # Для DatasetDict: токенизируем только целевой сплит
        tokenized_dataset = dataset[target_split].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=columns_to_remove
        )
    else:
        # Для одного Dataset
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=columns_to_remove
        )
    
    # Разделение на train и validation
    logger.info(f"Разделение данных (test_size={TEST_SIZE})...")
    if len(tokenized_dataset) > 100:  # Только если данных достаточно
        split_dataset = tokenized_dataset.train_test_split(test_size=TEST_SIZE, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        logger.info(f"Размер обучающей выборки: {len(train_dataset)}")
        logger.info(f"Размер валидационной выборки: {len(eval_dataset)}")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        logger.warning(f"Мало данных ({len(tokenized_dataset)} примеров). Валидация не будет использоваться.")
    
    # Загрузка модели
    logger.info(f"Загрузка модели {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Обновление training_args с output_dir
    training_args.output_dir = output_dir
    
    # Создание Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Обучение
    logger.info("Начинаю обучение...")
    train_result = trainer.train()
    
    # Сохранение метрик
    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    
    if eval_dataset is not None:
        logger.info("Оценка на валидационном наборе...")
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
    
    # Сохранение модели
    logger.info(f"Сохранение модели в {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Очистка
    cleanup_dataset(dataset)
    cleanup_dataset(tokenized_dataset)
    
    return output_dir

def main():
    """Основная функция последовательного обучения."""
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПОСЛЕДОВАТЕЛЬНОГО ДООБУЧЕНИЯ МОДЕЛИ")
    logger.info(f"Базовая модель: {BASE_MODEL_NAME}")
    logger.info(f"Количество датасетов: {len(DATASET_NAMES)}")
    logger.info("=" * 60)
    
    current_model = BASE_MODEL_NAME
    
    for i, dataset_name in enumerate(DATASET_NAMES, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ДАТАСЕТ {i}/{len(DATASET_NAMES)}: {dataset_name}")
        logger.info(f"Текущая модель: {current_model}")
        logger.info(f"{'='*60}")
        
        # Создание имени для сохранения модели
        # Заменяем "/" на "-" в названии датасета
        safe_dataset_name = dataset_name.replace("/", "-")
        
        # Если это первый датасет, используем базовое имя модели
        if current_model == BASE_MODEL_NAME:
            model_base_name = os.path.basename(BASE_MODEL_NAME)
        else:
            # Иначе извлекаем имя из пути
            model_base_name = os.path.basename(current_model)
        
        # Формируем имя выходной директории
        output_dir = f"{model_base_name}_{safe_dataset_name}"
        
        # Создаем training_args для текущего датасета
        training_args = Seq2SeqTrainingArguments(**TRAINING_CONFIG)
        
        # Обучаем на датасете
        new_model_path = train_on_dataset(
            model_name=current_model,
            dataset_name=dataset_name,
            output_dir=output_dir,
            training_args=training_args
        )
        
        if new_model_path:
            current_model = new_model_path
            logger.info(f"Модель успешно сохранена как: {new_model_path}")
            
            # Сохраняем информацию о процессе
            with open("training_log.txt", "a") as f:
                f.write(f"Датасет {i}: {dataset_name}\n")
                f.write(f"Результирующая модель: {new_model_path}\n")
                f.write("-" * 50 + "\n")
        else:
            logger.error(f"Обучение на датасете {dataset_name} завершилось с ошибкой")
            break
    
    logger.info("\n" + "=" * 60)
    logger.info("ПОСЛЕДОВАТЕЛЬНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    logger.info(f"Финальная модель сохранена в: {current_model}")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Проверка доступности GPU
    if torch.cuda.is_available():
        logger.info(f"Найдено GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("GPU не обнаружен, используется CPU")
    
    main()
