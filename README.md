This is the official repo of our paper [PerturbScore: Connecting Discrete and Continuous Perturbations in NLP](https://arxiv.org/abs/2310.08889) to appear in EMNLP 2023 findings, we are hoping that the concept of studying discrete perturbations in NLP through building the connections between continuous perturbations can provide hints for future studies. The repo contains:

- The [data](https://github.com/renke999/PerturbScore/tree/main/meta_data) used for training the PerturbScore.
- The code for [training the PerturbScore](https://github.com/renke999/PerturbScore/tree/main/meta_data).
- The code for [generating the data](https://github.com/renke999/PerturbScore/tree/main/scripts).

For technical details and additional experimental results, please refer to our paper.



## Data Release

We release our PerturbScore dataset under the [meta_data](https://github.com/renke999/PerturbScore/tree/main/meta_data) folder. The `perprocessed_data.csv` is the raw data file, we then randomly select 80% data tuples as the training set(`train_data.csv`) and 20% as the test set(`test_data.csv`). These CSV files contain the following fields:

- raw: `str`, The raw sentence in the dataset.
- attacked: `str`, The sentence after the discrete attack using textattack.
- epsilon: `float`， The continuous max norm ball $\epsilon$.
- timestep: `int`, The number of attack steps in the discrete attack.
- cont_sim: `float`, $cos(f_\theta(X+\delta), f_\theta(X))$
- disc_sim: `float`, $cos(f_\theta(S+P(S)), f_\theta(S))$
- difference: `float`, Measuring the cosine similarity difference between discrete and continuous perturbations.
- model_input: `str`, The  sentence constructed as input for training PerturbScore.
- bert_score_precision: `float`, the precision of $BertScore(S+P(S), S)$
- bert_score_recall: `float`, the recall of $BertScore(S+P(S), S)$
- bert_score_f1: `float`, the f1 score of $BertScore(S+P(S), S)$
- use_sim: `float`, the universal sentence encoder output of $USE(S+P(S),S)$

The naming rule for the subfolders is `{dataset_model_discrete-attack}`:

- dataset: `['imdb', 'agnews']`, the dataset used to train bert.
- Model: `['bert', 'freelb']`, `bert` is the finetuned bert model while `freelb` is the adversially trained bert model using the `FreeLB` method.
- discrete-attack: `['rand', 'textfooler']`, `rand` is the random attack method while `textfooler` is the `textfooler` attack method. 



## PerturbScore Training And Evaluation

We use the released data above to train our PerturbScore. Specifically, we use `model_input` and `epsilon`field to train our PerturbScore, other fields are just for data integrity and reliability. 

To reproduce our training process, first install the requirements

```bash
pip install -r requirements.txt
```

Then run the  `scripts/run_meta_net.sh` script to train PerturbScore on the `imdb_bert_rand` dataset, Feel free to uncomment the script to train other models you want. This script will help you train and evaluate the model.

```bash
./scripts/run_meta_net.sh
```

If you want to evaluate the trained PerturbScorer against a given dataset, run the `scripts/run_evaluation.sh` script where you should specify the model and dataset path. The script will output the Kendall's tau and Spearman's (rho) rank correlation coefficient between our PerturbScorer and max norm range Epsilon, it can also help you evaluate the cross-pertrubation/dataset/model generalization ability of our PerturbScorer. 
```bash
./scripts/run_evaluation.sh
```


## Data Generation Process

Data Generation is a non-trivial process owing to the huge search space and multiple datasets and discrete attacking methods. We release the code for finetuning bert, extracting features from continuous attacks, extracting features from discrete attacks, and we use a grid search implementation to generate the data for efficiency and simlicity. See [scripts/README.md](https://github.com/renke999/PerturbScore/blob/main/scripts/README.md) for more details.
