#!/usr/bin/env bash

function runexp {

export DATASET_NAME=${1}

gpu=${2}      # The GPU you want to use
mname=${3}    # Model name
alr=${4}      # Step size of gradient ascent
amag=${5}     # Magnitude of initial (adversarial?) perturbation
anorm=${6}    # Maximum norm of adversarial perturbation
asteps=${7}   # Number of gradient ascent steps for the adversary
lr=${8}       # Learning rate for model parameters
bsize=${9}    # Batch size
gas=${10}     # Gradient accumulation. bsize * gas = effective batch size
seqlen=128    # Maximum sequence length
hdp=${11}     # Hidden layer dropouts for ALBERT
adp=${12}     # Attention dropouts for ALBERT
ts=${13}      # Number of training steps (counted as parameter updates)
ws=${14}      # Learning rate warm-up steps
seed=${15}    # Seed for randomness
wd=${16}      # Weight decay

# 1. if adv_steps=0  ->  we are training bert on ag_news
# 2. if adv_steps!=0  ->  we are adv traing bert using freelb on ag_news
if [ "${asteps}" -eq 0 ]; then
  expname="${DATASET_NAME}-bert-base-uncased-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-hdp${hdp}-adp${adp}-ts${ts}-ws${ws}-wd${wd}-seed${seed}"
else
  expname="${DATASET_NAME}-FreeLB-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-hdp${hdp}-adp${adp}-ts${ts}-ws${ws}-wd${wd}-seed${seed}"
fi


CUDA_VISIBLE_DEVICES=${gpu} python ../model/freelb.py \
  --model_name_or_path ${mname} \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length ${seqlen} \
  --pad_to_max_length False \
  --per_device_train_batch_size ${bsize} \
  --gradient_accumulation_steps ${gas} \
  --learning_rate ${lr} --weight_decay ${wd} \
  --gpu_id ${gpu} \
  --output_dir outputs/${expname}/ \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
  --adv_lr ${alr} --adv_init_mag ${amag} --adv_max_norm ${anorm} --adv_steps ${asteps} \
  --max_steps ${ts} --warmup_steps ${ws} --seed ${seed} \
  --logging_steps 100 --save_steps 100 \
  --evaluation_strategy epoch
}


# runexp DATASET_NAME  gpu      model_name       adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp      ts     ws     seed      wd
# 1. freelb on ag_news (adv_steps=3)
runexp  ag_news       1       bert-base-uncased   1e-1    6e-1      0    3    2e-5     32       1        0.1   0    18750   1875     42     1e-2

# 2. bert on ag_news (adv_steps=0)
runexp  ag_news       2       bert-base-uncased   1e-1    6e-1      0    0    2e-5     32       1        0.1   0    18750   1875     42     1e-2
