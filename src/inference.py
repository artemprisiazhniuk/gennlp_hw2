import unsloth
from unsloth import FastLanguageModel
import torch
import argparse


def load_model(model_path: str, max_seq_length: int = 2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@torch.inference_mode()
def generate_poem(
    model,
    tokenizer,
    topic: str,
    max_new_tokens: int = 256,
    temperature: float = 0.9
):
    messages = [
        {
            "role": "system", 
            "content": "Ты полезный ассистент и помогаешь отвечать на разные вопросы. Отвечай в 1-2 предложения."
        },
        {
            "role": "user",
            "content": topic
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to saved LoRA adapter directory")
    parser.add_argument("--author", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, tokenizer = load_model(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )

    result = generate_poem(
        model=model,
        tokenizer=tokenizer,
        topic=args.topic,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(result)


if __name__ == "__main__":
    main()