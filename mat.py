# test_minimal.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# Минимальный тестовый датасет
data = {
    'problem': ['2+2', '3*3', '10/2'],
    'solution': ['4', '9', '5']
}
dataset = Dataset.from_dict(data)

def create_prompt(ex):
    return {'input_text': f'Реши задачу: {ex["problem"]}', 'target_text': ex['solution']}

dataset = dataset.map(create_prompt)

# Токенизация
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
def tokenize_fn(ex):
    inputs = tokenizer(ex['input_text'], truncation=True, padding='max_length', max_length=50)
    labels = tokenizer(ex['target_text'], truncation=True, padding='max_length', max_length=20)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized = dataset.map(tokenize_fn, batched=True)

# Разделение
split = tokenized.train_test_split(test_size=0.2)

# Модель
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Аргументы
args = Seq2SeqTrainingArguments(
    output_dir='./test_output',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_strategy='no'
)

# Trainer - без tokenizer параметра
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=split['train'],
    eval_dataset=split['test'],
    data_collator=data_collator,
)
trainer.tokenizer = tokenizer  # Устанавливаем отдельно

# Краткое обучение
trainer.train()
