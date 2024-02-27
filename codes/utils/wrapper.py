from datetime import datetime

from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel, PeftConfig

class LanguageModelWrapper:
    def __init__(
            self,
            model_name,
            use_4bit=True,
            max_memory_map=None,
            max_new_tokens=256,
            eos_token_id=0,
            do_sample=True,
        ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.max_memory_map = max_memory_map
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.do_sample = do_sample

        if model_name is not None:
            if use_4bit:
                self.q_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else: # use 8bit
                self.q_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=self.q_config,
                trust_remote_code=True,
                max_memory=max_memory_map,
            )

            self.model.eval()
            self.model.config.use_cache = True

    def get_model_name(self):
        return self.model_name

    def generate_in_stream(self, input_text):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **self.tokenizer(
                input_text,
                return_tensors="pt",
                return_token_type_ids=False,
            ),
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            do_sample=self.do_sample,
        )

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        thread.start()

        tokens = []
        for new_token in streamer:
            tokens.append(new_token)
            yield new_token

        thread.join()
    def generate(self, input_text):
        generation_result = self.model.generate(
            **self.tokenizer(
                input_text, 
                return_tensors='pt', 
                return_token_type_ids=False
            ).to('cuda'), 
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
            eos_token_id=self.eos_token_id,
            do_sample=self.do_sample,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )
        generated_text = self.tokenizer.decode(generation_result[0])
        slide_len = len(input_text)
        
        return generated_text, slide_len

class LoRAWrapper(LanguageModelWrapper):
    def __init__(
            self,
            checkpoint_dir_path,
            use_4bit=True,
            max_memory_map=None,
            max_new_tokens=64,
            eos_token_id=3,
            do_sample=False,
        ):
        self.checkpoint_dir_path = checkpoint_dir_path
        super().__init__(
            model_name=None,
            use_4bit=use_4bit,
            max_memory_map=max_memory_map,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
        )

        self.plm_config = PeftConfig.from_pretrained(checkpoint_dir_path)
        if use_4bit:
            self.q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else: # use 8bit
            self.q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_config.base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                self.plm_config.base_model_name_or_path,
                device_map="auto",
                quantization_config=self.q_config,
                trust_remote_code=True,
                max_memory=max_memory_map,
            ),
            checkpoint_dir_path,
        )

        self.model.eval()
        self.model.config.use_cache = True

    def get_model_name(self):
        return self.checkpoint_dir_path