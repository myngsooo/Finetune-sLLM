# Finetune-sLLM for RAG
This repo contains a implementation of fine-tuning sLLM for RAG, based on [open source code](https://github.com/kh-kim/sllm_finetune).

## Requirements

First, run the following script to install the remaining dependencies.

```bash
pip install -r requirements.txt
```

### Download the training/dev dataset
We use the [WebGLM-QA](https://huggingface.co/datasets/THUDM/webglm-qa) dataset, but if you prefer, you can explore alternative options.

| **WebGLM-QA** | **train** | **dev** | **test** |
|:--------:|:--------:|:--------:|:--------:|
| **num** | 43.6k | 1k | 0.4k |

```bash
cd ./codes/data
bash download
```

## Train

```bash
bash run_train.sh
```
or
```bash
INPUT_FN=''
OUTPUT_DIR_OR_PATH='./result'
MODEL_NAME_OR_PATH='google/gemma-7b-it' 

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
```

## Inference
You can run the commands below for inference after using the repo to train a model:

```bash
bash run_eval.sh
```
or
```bash
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
```

## Result

```
{"pred":"### Answer:When a Canadian mortgage is up for renewal and the current contract term has expired, a lender is required to provide the homeowner with an option to choose subsequent payment terms[2]. This means the lender has the option of offering the same terms as before, negotiating new terms, and\/or declining to offer a renewal[3",
  "gold":"When renewing a mortgage in Canada, your lender must notify you in advance of the renewal date with your options for renewal terms[1][2]. Your mortgage will typically automatically renew or become in default if you don't take action[3]. Depending on your lender, you may be able to renew your mortgage as early as 6 months prior to your current mortgage term expiring[2][3][5]. RBC Royal Bank mortgage customers can choose Subsequent Payment Terms and be protected from an increase in interest rates for the interest type and term they selected[4]."}
{"pred":"### Answer:Driving into moderate to severe snowfall can be dis orienting because it makes visibility difficult. The snowflakes reflect the light of your car's headlights, obstructing your vision and making it difficult for you to navigate safely. Additionally, heavy rain can make it hard to find your way, as it can obstruct visibility and make the road",
  "gold":"Driving into mild to heavy snowfall at night can be disorienting because the snow can make it difficult to see and the headlights can reflect off of the snowflakes in the air, obstructing the view[1]. Additionally, snow falling heavily can be so disorienting that it can make it hard to determine where you are going. The darkness of a snowless roadway can also contribute to the disorientation, as it can blend into the dark sky and create a sudden contrast when the snow begins to fall[5]."}
{"pred":"### Answer:If a women with milk allergies starts to lactate, she may experience an allergy reaction to her milk due to cow’s or soy formula that she is using to feed her baby[1]. This can range in severity from wheezes to digestive issues and anaphylactic shock[2]. If the woman is breastfeeding, the allergy",
  "gold":"If a woman with a serious milk allergy starts lactating, she may experience an allergic reaction to the milk proteins that are passed through her breast milk. This can result in symptoms such as wheezing, vomiting, hives, and digestive problems[2], and can even cause anaphylaxis—a severe, life-threatening reaction[2]. The reaction can occur within minutes of latching, pumping, or hand expressing, or even after 30+ minutes[5]. The woman may need to exclude all products that contain milk from her diet in order to prevent an allergic reaction[3]."}
```
