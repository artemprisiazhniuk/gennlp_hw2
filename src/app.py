import os
from functools import partial
import gradio as gr
import argparse
import yaml

import torch
from unsloth import FastLanguageModel

from inference import *


def chat(query, model, tokenizer):
    if not query.strip():
        return "Введите вопрос"
    
    answer = generate_poem(
        model=model,
        tokenizer=tokenizer,
        topic=query,
    )

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config-path", default="config")
    parser.add_argument("--config", default="config.yaml")
    
    parser.add_argument("--checkpoint-path", default="checkpoints")
    parser.add_argument("--checkpoint", required=True)
    
    args = parser.parse_args()
    
    with open(os.path.join(args.config_path, args.config), "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir = os.path.join(args.checkpoint_path, args.checkpoint)
    
    
    model, tokenizer = load_model(
        model_path=output_dir
    )
    
    with gr.Blocks() as demo:
        gr.Markdown("## Style Transfer Chatbot")

        with gr.Row():
            query_input = gr.Textbox(label="Ваш запрос")

        answer_output = gr.Textbox(label="Ответ")

        submit_btn = gr.Button("Ответить")

        submit_btn.click(
            lambda x: chat(x, model, tokenizer),
            inputs=[query_input],
            outputs=[answer_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)