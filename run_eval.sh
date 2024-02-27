CHECK_POINT=''

python ./src/inference.py \
    --checkpoint_dir_path ${CHECK_POINT} \
    --use_4bit \
    --eos_token_id 3 \
    --max_new_tokens 128 \

