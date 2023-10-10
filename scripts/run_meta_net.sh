#!/usr/bin/env bash

# 8000 sample, bsize 16, 8000/16=500 --> 1 epoch 500 steps

function runexp {
  gpu=${1}
  type=${2}
  max_len=${3}

  exp_name=meta_net_${type}

  CUDA_VISIBLE_DEVICES=${gpu} python ../model/meta_net.py \
    --model_name_or_path bert-base-uncased \
    --train_file ../meta_data/${type}/train_data.csv \
    --validation_file ../meta_data/${type}/eval_data.csv \
    --do_train \
    --do_eval \
    --max_seq_length ${max_len} \
    --pad_to_max_length False \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --logging_steps 200 --save_steps 200 \
    --evaluation_strategy epoch \
    --output_dir outputs/${exp_name}/ \
    --overwrite_output_dir
}

#runexp    gpu    type                        max_len
runexp     1      imdb_bert_rand              512
#runexp     1      imdb_bert_textfooler        512
#runexp     1      imdb_freelb_rand              512
#runexp     1      imdb_freelb_textfooler        512
#runexp     1      agnews_bert_rand            128
#runexp     1      agnews_bert_textfooler      128
#runexp     1      agnews_freelb_rand          128
#runexp     1      agnews_freelb_textfooler    128