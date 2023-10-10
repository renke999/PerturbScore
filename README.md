This is the repo for the PerturbScore paper, we are hoping that the concept of studying discrete perturbations in NLP through building the connections between continuous perturbations can provide hints for future studies. The repo contains:

- The [data](https://github.com/renke999/PerturbScore/tree/main/meta_data) used for training the PerturbScore.
- The code for [training the PerturbScore](https://github.com/renke999/PerturbScore/tree/main/meta_data).
- The code for [generating the data](https://github.com/renke999/PerturbScore/tree/main/scripts).

For technical details and additional experimental results, please refer to our paper:

PerturbScore: Connecting Discrete and Continuous Perturbations in NLP. In the Findings of EMNLP 2023



## Data release

We release our PerturbScore dataset under the `meta_data` folder. The `perprocessed_data.csv` is the raw data file, we then ramdomly select 80% data tuples as the training set(`train_data.csv`) and 20% as the test set(`test_data.csv`). These CSV files contain the following fields:

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

- dataset: `['imdb', 'agnews']`, which is the dataset used to train bert.
- Model: `['bert', 'freelb']`, `bert` is the finetuned bert model whie `freelb` is the adversially trained bert model using `FreeLB` method.
- discrete-attack: `['rand', 'textfooler']`, `rand` is the random attack method while `textfooler` is the `textfooler` attack method. 



## PerturbScore Training

We use the released data above to train our PerturbScore. Specifically, we use `model_input` and `epsilon`field to train our PerturbScore, other filelds are just for data integrity and reliability. 

To reproduce our training process, first install the requirements

```bash
pip install -r requirements.txt
```

Then run the  `scripts/run_meta_net.sh` script to train PerturbScore on `imdb_bert_rand` dataset, Feel free to uncomment the script to train other models you want.

```bash
./scripts/run_meta_net.sh
```



## Data Generation Process

Data Generation is a non-trival process owing to the huge searching space and multiple datasets and discrete attacking methods. We release the code for finetuning bert, extracting features from continous attacks, extracting features from discrete attacks, and we use a grid search implementation to generate the data for efficiency and simlicity. See [scripts/README.md](https://github.com/renke999/PerturbScore/blob/main/scripts/README.md) for more details.
