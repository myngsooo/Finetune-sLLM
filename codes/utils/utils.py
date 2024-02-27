import json
import random
import datasets

from prompt import basic_prompt

def get_datasets(input_fn, valid_ratio=0.05, valid_fn=None):
    data = get_json(input_fn)
    random.shuffle(data)
    print(f"Sample data:\n{data[0]}")

    train_data = data[:int(len(data) * (1 - valid_ratio))]
    valid_data = data[int(len(data) * (1 - valid_ratio)):]

    train_dataset = datasets.Dataset.from_dict({"text": train_data})
    valid_dataset = datasets.Dataset.from_dict({"text": valid_data})

    return train_dataset, valid_dataset

def get_json(fn, prompt=basic_prompt()):
    return [prompt.format(**json.loads(line)) for line in open(fn, "r")]