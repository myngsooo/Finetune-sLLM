import json
import random
import datasets

from utils.prompt import get_prompt

def get_datasets(input_fn, valid_ratio=0.05, valid_fn=None):
    data, _ = get_json(input_fn, mode='train')
    random.shuffle(data)
    print(f"Sample data:\n{data[0]}")

    train_data = data[:int(len(data) * (1 - valid_ratio))]
    valid_data = data[int(len(data) * (1 - valid_ratio)):]

    train_dataset = datasets.Dataset.from_dict({"text": train_data})
    valid_dataset = datasets.Dataset.from_dict({"text": valid_data})

    return train_dataset, valid_dataset


def get_json(fn, mode='test'):
    data, answers = [], []
    with open(fn, "r") as f:
        for line in f:
            js = json.loads(line)
            if mode=='test':
                prompt=get_prompt(mode='test')
                context = js["references"]
                query = js["question"]
                data.append(prompt.format(
                    context=context,
                    query=query,
                ))
                answers.append(js["answer"])
            elif mode=='train':
                prompt=get_prompt(mode='train')
                context = js["references"]
                query = js["question"]
                answer = js["answer"]
            
                data.append(prompt.format(
                    context=context,
                    query=query,
                    answer=answer,
                ))
    return data, answers