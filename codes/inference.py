import argparse

from utils.wrapper import LoRAWrapper
from utils.prompt import prompt_test
    
def get_config():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint_dir_path", 
        type=str, 
        required=True,
    )
    p.add_argument(
        "--use_4bit", 
        action="store_true"
    )
    p.add_argument(
        "--eos_token_id", 
        type=int,
        default=3,
    )
    p.add_argument(
        "--max_new_tokens", 
        type=int,
        default=128,
    )        
    config = p.parse_args()

    return config

def main(config):
    
    lora_wrapper = LoRAWrapper(
    checkpoint_dir_path=config.checkpoint_dir_path,
    use_4bit=config.use_4bit,
    eos_token_id=config.eos_token_id,
)
    
    input_text =prompt_test()

    print(lora_wrapper.generate(input_text))

if __name__ == "__main__":
    config = get_config()
    main(config)