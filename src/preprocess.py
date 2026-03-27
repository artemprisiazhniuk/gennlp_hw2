import os
import re
import argparse
from datasets import load_dataset
from openai import OpenAI
import dotenv


dotenv.load_dotenv('.env')

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.aitunnel.ru/v1/"
)


def add_example(x, text_field):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Сгенерируй простой вопрос, на который можно ответить следующим текстом."
                "Вопрос должен быть коротким (до 15 слов) и затрагивать только одну тему"
                "Верни только вопрос, не добавляй и не предлагай ничего больше."},
            {"role": "user", "content": x[text_field]}
        ]
    )

    try:
        question = response.choices[0].message.content
    except Exception as e:
        question = ', '.join(eval(x['keywords']))
    
    
    return {
        "messages": [
            {"role": "user", "content": f"Автор: {x['author']}\n Тема: {question}"},
            {"role": "assistant", "content": x[text_field].strip()}
        ]
    }


def format_example(example, tokenizer=None):
    return {
        "example": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }


def preprocess(args):
    raw_dataset = load_dataset(args.dataset_name, "default", split="train")
    
    ds = raw_dataset.filter(lambda item: 
            (item["author"] == 'Пушкин') and \
            (len(item["text"].split()) > args.min_words) and \
            (len(item["text"].split('\n')) > args.min_lines)
        )
    
    ds = ds.train_test_split(
        min(len(ds), args.limit / len(ds))
        )['test']
    
    ds = ds.map(lambda x: add_example(x, args.text_field), num_proc=4)
    ds.to_json(args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset-name", default="AnyaSchen/russian_poetry_with_keywords")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-lines", default=5)
    parser.add_argument("--min-words", default=5)
    parser.add_argument("--limit", default=500)
    
    parser.add_argument("--data-path", default="data/train.jsonl")
    
    args = parser.parse_args()
    
    preprocess(args)
            
    print("Сохранено в:", args.data_path)
        