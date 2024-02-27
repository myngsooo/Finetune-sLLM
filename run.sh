INPUT_FN=''
OUTPUT_DIR_OR_PATH='./result'
MODEL_NAME_OR_PATH='google/gemma-7b' 

python ./codes/train.py \
    --input_fn ${INPUT_FN} \
    --output_dir_path ${OUTPUT_DIR_OR_PATH} \
    --pretrained_model_name ${MODEL_NAME_OR_PATH} \
    --valid_ratio 0.025 \
    --max_length 2048 \
    --num_train_epochs 3 \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --save_total_limit 3 \
    --min_warmup_steps 200 \
    --warmup_ratio 0.1 \
    --num_logging_steps_per_epoch 1000 \
    --num_eval_steps_per_epoch 5 \
    --num_save_steps_per_epoch 5 \
    --use_4bit \
    --lora_r 4 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
