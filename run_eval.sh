INPUT_FN=''
CHECK_POINT=''
OUTPUT_DIR=''

python ./codes/inference.py \
    --input_fn ${INPUT_FN} \
    --checkpoint_dir_path ${CHECK_POINT} \
    --output_dir ${OUTPUT_DIR} \
    --use_4bit \
    --eos_token_id 3 \
    --max_new_tokens 128 \

