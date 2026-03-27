import unsloth
from unsloth import FastLanguageModel

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)

import os
import yaml
from functools import partial
import argparse


def build_model(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype = torch.float16
    )
    
    method = cfg["method"].lower()

    if method in ["prompt", "both"]:
        prompt_cfg = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=int(cfg["prompt_num_virtual_tokens"]),
            prompt_tuning_init=(
                PromptTuningInit.TEXT
                if cfg.get("prompt_init_text")
                else PromptTuningInit.RANDOM
            ),
            prompt_tuning_init_text=cfg.get("prompt_init_text", None),
            tokenizer_name_or_path=cfg["model_name"] if cfg.get("prompt_init_text") else None,
        )
        model = get_peft_model(model, prompt_cfg)

    if method in ["lora", "both"]:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            target_modules=cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )

    return model, tokenizer


def format_example(example, tokenizer=None, system_prompt=None):
    messages = example["messages"]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    result = {
        "example": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }
    
    return result
    

def main(cfg):
    model, tokenizer = build_model(cfg)

    dataset = load_dataset("json", data_files=cfg["data_path"])["train"]
    dataset = dataset.map(partial(format_example, tokenizer=tokenizer, system_prompt=cfg.get("system_prompt")))

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=int(cfg["batch_size"]),
            gradient_accumulation_steps=int(cfg["grad_accum"]),
            learning_rate=float(cfg["lr"]),
            warmup_steps=int(cfg["warmup_steps"]),
            lr_scheduler_type=cfg["lr_scheduler_type"],
            num_train_epochs=int(cfg["epochs"]),
            logging_steps=int(cfg["logging_steps"]),
            save_steps=int(cfg["save_steps"]),
            fp16=True,
            bf16=False,
            optim="adamw_8bit",
            max_length=int(cfg["max_seq_length"]),
            report_to="tensorboard",
        ),
        dataset_text_field=cfg["dataset_text_field"],
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config-path", default="config/")
    parser.add_argument("--config", default="config.yaml")
    
    parser.add_argument("--output-path", default="outputs/")
    
    parser.add_argument("--method", choices=["lora", "prompt", "both"], default=None)
    
    parser.add_argument("--data-path", default="data/train.jsonl")
    
    parser.add_argument("--logging-steps", default=10)
    parser.add_argument("--save-steps", default=100)
    
    args = parser.parse_args()
    
    run_name = args.config.split(".")[0]
    if args.method is not None:
        run_name = f"{run_name}_{args.method}"
    
    args.output_path = os.path.join(args.output_path, run_name)
    
    with open(os.path.join(args.config_path, args.config), "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["data_path"] = args.data_path
    cfg["output_dir"] = args.output_path
    cfg["logging_steps"] = args.logging_steps
    cfg["save_steps"] = args.save_steps
    
    if args.method is not None:
        cfg["method"] = args.method
        
    main(cfg)