import os
import argparse
import re
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from utils.utils import get_datasets

def get_config():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_fn", 
        required=True
    )
    p.add_argument(
        "--output_dir_path",
        default="./checkpoints"
    )
    p.add_argument(
        "--pretrained_model_name", 
        default="google/gemma-7b"
    )
    p.add_argument(
        "--valid_ratio", 
        type=float, 
        default=0.025
    )
    p.add_argument(
        "--max_length", 
        type=int, 
        default=2048
    )
    p.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1
    )
    p.add_argument(
        "--batch_size_per_device", 
        type=int, 
        default=4
    )
    p.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4
    )
    p.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5
    )
    p.add_argument(
        "--save_total_limit", 
        type=int, 
        default=3
    )
    p.add_argument(
        "--min_warmup_steps", 
        type=int, 
        default=1000
    )
    p.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.1
    )
    p.add_argument(
        "--num_logging_steps_per_epoch", 
        type=int, 
        default=1000
    )
    p.add_argument(
        "--num_eval_steps_per_epoch", 
        type=int, 
        default=5
    )
    p.add_argument(
        "--num_save_steps_per_epoch", 
        type=int, 
        default=5
    )
    p.add_argument(
        "--use_8bit", 
        action="store_true"
    )
    p.add_argument(
        "--use_4bit", 
        action="store_true"
    )
    p.add_argument(
        "--lora_r", 
        type=int, 
        default=4
    )
    p.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32
    )
    p.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05
    )
    config = p.parse_args()

    return config

def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def main(config):
    assert config.use_8bit ^ config.use_4bit, "You can only use one of 8bit and 4bit quantization."

    train_dataset, valid_dataset = get_datasets(
        config.input_fn,
        valid_ratio=config.valid_ratio,
    )

    print('|Train| =', len(train_dataset))
    print('|Eval| =', len(valid_dataset))

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_length), batched=True)
    valid_dataset = valid_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_length), batched=True)

    # Get BitsAndBytesConfig for quantization
    if config.use_8bit: 
        q_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif config.use_4bit:
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        q_config = None

    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        quantization_config=q_config, 
        use_cache = False,
        use_flash_attention_2=False,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    l_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        # "embed_tokens", "wte", "lm_head" 임베딩 레이어 및 아웃풋 레이어
        # "q_proj", "k_proj", "v_proj", "o_proj" 어텐션 연산이 이루어지는 레이어의 집합, o_proj는 self attention 출력 단계이기에 같이 pairing
        # "gate_proj", "down_proj", "up_proj" feed forward 네트워크 내에서 사용되는 모듈중 gate 메커니즘
        # "fc_in", "fc_out" feed forward 네트워크 중 fully connected하게 입력을 받는 층과 출력을 내보내는 곳을 같이 페어링
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, l_config)

    gpu_count = torch.cuda.device_count()
    total_batch_size = config.batch_size_per_device * gpu_count
    num_iterations_per_epoch = int((len(train_dataset) / total_batch_size) / config.gradient_accumulation_steps)
    logging_steps = max(10, int(num_iterations_per_epoch / config.num_logging_steps_per_epoch))
    eval_steps = max(10, int(num_iterations_per_epoch / config.num_eval_steps_per_epoch))
    save_steps = max(10, int(num_iterations_per_epoch / config.num_save_steps_per_epoch))
    warmup_steps = max(
        config.min_warmup_steps,
        num_iterations_per_epoch * config.num_train_epochs * config.warmup_ratio,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        half_precision_backend="auto",
        bf16=True,    # If you want to use fp16, change it to fp16.
        bf16_full_eval=True,    # If you want to use fp16_full_eval, change it to fp16_full_eval.
        optim="paged_adamw_8bit"
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(config.output_dir_path)

if __name__ == "__main__":
    config = get_config()
    main(config)
