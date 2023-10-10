# random attack 
CUDA_VISIBLE_DEVICES=7 python -m textattack attack \
    --model-from-huggingface {{BERT_MODEL_PATH}} \
    --dataset-from-huggingface ag_news \
    --recipe random1225 \
    --num-examples 1000 \
    --training_dataset ag_news \
    --save_path {{SAVE_PATH}}
    
# textfooler attack
CUDA_VISIBLE_DEVICES=7 python -m textattack attack \
    --model-from-huggingface {{BERT_MODEL_PATH}} \
    --dataset-from-huggingface ag_news \
    --recipe textfooler \
    --num-examples 1000 \
    --training_dataset imdb \
    --save_path {{SAVE_PATH}}
