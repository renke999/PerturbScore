This folder contains shell scripts that finetune bert, generate data and train our PerturbScore. 

For the imdb dataset:

1. Run `run_imdb.sh` to (adversailly) normally train a bert model and adversially train a model using FreeLB method.

   - You need to specify the path for saving the model in the scripts.

2. Run `run_hpp_freelb_imdb.sh` and `run_hpp_bert_imdb.sh` to extract the [CLS] embedding before and after the continous PGD attack. 

   - You need to specify the bert model path trained before and the tensor saving path in the scripts. 

   - The saved tensor is a list of dictionaries which have the keys:

     ```
     [{'input_ids', 'label', 'timestep', 'layer_0', 'layer_1', ..., 'layer_13'},
      {'input_ids', 'label', 'timestep', 'layer_0', 'layer_1', ..., 'layer_13'},
      {'input_ids', 'label', 'timestep', 'layer_0', 'layer_1', ..., 'layer_13'},...]
     ```

     - `input_ids`：`List(int)`, the input_ids of the sentence from huggingface tokenizer
     - `timestep`: `int`, 0 means it's the raw sentence while 1 means it's the attacked sentence
     - `label`: `int`, the predicted label of the raw or attacked sentence
     - `layer_i`:`tensor`, the embedding of the [CLS] token

   - Warning: this step consumes a lot of storage because we save the [CLS] embedding of all 13 bert layers for some preliminary experiment while the last layer alone is enough

3. We use `run_disc_freelb_imdb.sh` and `run_disc_bert_imdb.sh` to obtain the [CLS] embedding before and after the discrete random and textfooler attack. 

   - You need to specify the bert model path trained before and the tensor saving path in the scripts.
   - The saved tensor has the same shape and fields above.

    > We implement discrete attack based on the [TextAttack](https://github.com/QData/TextAttack) package. Specifically, we add `probe.py`, `random_swap_20221225.py, ` modify `greedy_word_swap_wir.py`, `huggingface_dataset.py` and some other files to support our custom attack methods and command line arguments.
   >
   > To use our implementation, first run the following command.
   >
   > ```bash
   > cd TextAttack
   > pip install -e .
   > ```
   >
   > Then you can run our shell scripts to reproduce the attacking process and extracting features.
   >
   > ```bash
   > ./run_disc_freelb_imdb.sh
   > ./run_disc_bert_imdb.sh
   > ```

4. We use grid search to match continuous and discrete [CLS] embedding, which results in our csv files under the `meta_data` folder. Grid search is the same implementation as the paper, only to be more efficient and simple. Grid search means that we extracted $cos(f_\theta(X+\delta), f_\theta(X))$  of all the different continuous epsilons and $cos(f_\theta(S+P(S)), f_\theta(S))$ of all the different discrete attack time steps, then we try to find the most similar continuous and discrete sample pair. This process is non-trival and we provide a sample notebook `meta_data_imdb_bert.ipynb` for reference only.

5. Finally run `run_meta_net.sh` to train our PerturbScore.


The scripts for running ag_news dataset is similar to above in the `scripts_for_agnews` folder. 