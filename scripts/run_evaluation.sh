#!/usr/bin/env bash

# 8000 sample, bsize 16, 8000/16=500 --> 1 epoch 500 steps

function runexp {
  gpu=${1}
  model=${2}
  dataset=${3}
  max_len=${4}

  exp_name=evaluation_${dataset}

  CUDA_VISIBLE_DEVICES=${gpu} python ../model/meta_net.py \
    --model_name_or_path outputs/meta_net_${model} \
    --train_file ../meta_data/${dataset}/train_data.csv \
    --validation_file ../meta_data/${dataset}/eval_data.csv \
    --do_eval \
    --max_seq_length ${max_len} \
    --per_device_eval_batch_size 128 \
    --seed 42 \
    --include_inputs_for_metrics \
    --output_dir outputs/${exp_name}/
}



#runexp    gpu     model_path                 dataset                       max_len

########################## PerturbScorer evaluation ##########################
runexp     0      imdb_bert_rand             imdb_bert_rand                512
#runexp     0      imdb_freelb_rand           imdb_freelb_rand              512
#runexp     0      imdb_bert_textfooler       imdb_bert_textfooler          512
#runexp     0      imdb_freelb_textfooler     imdb_freelb_textfooler        512
#
#runexp     0      agnews_bert_rand           agnews_bert_rand              512
#runexp     0      agnews_freelb_rand         agnews_freelb_rand            512
#runexp     0      agnews_bert_textfooler     agnews_bert_textfooler        512
#runexp     0      agnews_freelb_textfooler   agnews_freelb_textfooler      512



########################## cross_perturbation evaluation ##########################
## 1. imdb-rand-bert
#runexp     0      imdb_bert_textfooler        imdb_bert_rand            512
#
## 2. imdb-rand-freelb
#runexp     0      imdb_freelb_textfooler      imdb_freelb_rand          512
#
## 3. imdb-textfooler-bert
#runexp     0      imdb_bert_rand      imdb_bert_textfooler          512
#
## 4. imdb-textfooler-freelb
#runexp     0      imdb_freelb_rand      imdb_freelb_textfooler          512



########################### cross_model evaluation ##########################
## 1. imdb-rand-bert
#runexp     0      imdb_freelb_rand        imdb_bert_rand            512
#
## 2. imdb-rand-freelb
#runexp     0      imdb_bert_rand      imdb_freelb_rand          512
#
## 3. imdb-textfooler-bert
#runexp     0      imdb_freelb_textfooler      imdb_bert_textfooler          512
#
## 4. imdb-textfooler-freelb
#runexp     0      imdb_bert_textfooler      imdb_freelb_textfooler          512



########################### cross_dataset evaluation ##########################
## 1. imdb-rand-bert
#runexp     0      agnews_bert_rand        imdb_bert_rand            512
#
## 2. imdb-rand-freelb
#runexp     0      agnews_freelb_rand      imdb_freelb_rand          512
#
## 3. imdb-textfooler-bert
#runexp     0      agnews_bert_textfooler      imdb_bert_textfooler          512
#
### 4. imdb-textfooler-freelb
#runexp     0      agnews_freelb_textfooler      imdb_freelb_textfooler          512