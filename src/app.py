import os
from functools import partial
import gradio as gr
import argparse
import yaml

import torch
from unsloth import FastLanguageModel


def chat(query, model, tokenizer):
    if not query.strip():
        return "Введите вопрос"
    
    # промпт
    prompt = f"Автор: Пушкин\nТема: {query.strip()}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config-path", default="config")
    parser.add_argument("--config", default="simple_sft.yaml")
    
    parser.add_argument("--checkpoint-path", default="outputs")
    parser.add_argument("--checkpoint", required=True)
    
    args = parser.parse_args()
    
    with open(os.path.join(args.config_path, args.config), "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir = os.path.join(args.checkpoint_path, args.checkpoint)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = output_dir,
        max_seq_length = cfg["max_seq_length"],
        dtype = torch.float16,
        load_in_4bit = cfg["load_in_4bit"]
    )
    
    FastLanguageModel.for_inference(model)
    
    with gr.Blocks() as demo:
        gr.Markdown("## Style Transfer Chatbot")

        with gr.Row():
            query_input = gr.Textbox(label="Ваш запрос")
            top_k_slider = gr.Slider(1, 10, value=args.top_k, step=1, label="Top-K")

        answer_output = gr.Textbox(label="Ответ")

        submit_btn = gr.Button("Ответить")

        submit_btn.click(
            lambda x: chat(x, model, tokenizer),
            inputs=[query_input],
            outputs=[answer_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)