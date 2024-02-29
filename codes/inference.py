import argparse
import datasets
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from evaluate import load

from utils.wrapper import LoRAWrapper
from utils.utils import get_json
from utils.prompt import get_prompt

def main(config):    
    lora_wrapper = LoRAWrapper(
    checkpoint_dir_path=config.checkpoint_dir_path,
    use_4bit=config.use_4bit,
    max_new_tokens=config.max_new_tokens,
    eos_token_id=config.eos_token_id,
    )
    test_data, gold_label = get_json(config.input_fn, mode='test')
    
    pred=[]
    for i in tqdm(range(len(test_data))):
        output = lora_wrapper.generate(test_data[i])
        idx = output.find("### Answer")
        pred.append(output[idx:].replace('\n', ''))
        data = pd.DataFrame({'pred': pred})
        data.to_json(config.output_dir, orient="records", lines=True, force_ascii = False)
    
    bertscore = load('bertscore')
    results = bertscore.compute(predictions=pred, references=gold_label, model_type="microsoft/deberta-xlarge-mnli")
    print(results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_fn", 
        type=str, 
        required=True,
    )
    p.add_argument(
        "--checkpoint_dir_path", 
        type=str, 
        required=True,
    )
    p.add_argument(
        "--output_dir", 
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
    main(config)
