import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import yaml
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import argparse


def main(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype = torch.float16
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )

    dataset = load_dataset("json", data_files=cfg["data_path"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=int(cfg["batch_size"]),
            gradient_accumulation_steps=int(cfg["grad_accum"]),
            learning_rate=float(cfg["lr"]),
            warmup_steps=int(cfg["warmup_steps"]),
            lr_scheduler_type=cfg["lr_scheduler_type"],
            num_train_epochs=int(cfg["epochs"]),
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            fp16=True,
            bf16=False,
            optim = "adamw_8bit",
            max_length=int(cfg["max_seq_length"]),
            report_to="tensorboard"
        )
    )

    trainer.train()

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config-path", default="config/")
    parser.add_argument("--config", default="config.yaml")
    
    parser.add_argument("--output-path", default="outputs/")
    
    parser.add_argument("--data-path", default="data/train.jsonl")
    
    parser.add_argument("--logging-steps", default=10)
    parser.add_argument("--save-steps", default=100)
    
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_path, args.config.split('.')[0])
    
    with open(os.path.join(args.config_path, args.config), "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["data_path"] = args.data_path
    cfg["output_dir"] = args.output_path
    cfg["logging_steps"] = args.logging_steps
    cfg["save_steps"] = args.save_steps
    
        
    main(cfg)