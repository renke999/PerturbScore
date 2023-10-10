import torch
import numpy as np

class FeatureProbe:
    """
    feature probe for bert_base(12 layer)
    """

    def __init__(self, save_path='freelb_tensor.pt', dataset='imdb'):
        """
        model: bert-base-uncased finetuned on imdb
        save_path: torch.save path
        """
        self.all_sentences = []
        self.save_path = save_path
        self.dataset = dataset

        self.use_encoder = None

    def get_save_path(self):
        return self.save_path

    def get_dataset(self):
        return self.dataset 

    def set_save_path(self, save_path):
        self.save_path = save_path 
        print()
        print("**********" * 5)
        print("initial save_path successfully!\nsave_path: {}".format(self.save_path))
        print("**********" * 5)
        print()

    def set_dataset(self, dataset):
        self.dataset = dataset
        print()
        print("**********" * 5)
        print("initial dataset successfully!\ndataset: {}".format(self.dataset))
        print("**********" * 5)
        print()

    def set_use_encoder(self, use_encoder):
        # class UniversalSentenceEncoder(SentenceEncoder):
        self.use_encoder = use_encoder
        print("initial use_encoder successfully!\n")

    # NOTE: 12.5 加了use_sim的代码
    def get_use_sim(self, raw_input, attacked_input):
        raw_embedding, attacked_embedding = self.use_encoder.encode(
            [raw_input, attacked_input]
        )
        # raw_embedding: 已由<class 'tensorflow.python.framework.ops.EagerTensor'>转换为numpy
        if not isinstance(raw_embedding, torch.Tensor):
            raw_embedding = torch.tensor(raw_embedding)

        if not isinstance(attacked_embedding, torch.Tensor):
            attacked_embedding = torch.tensor(attacked_embedding)

        # 为了满足torch的输入要求，增加一个batch维
        raw_embedding = torch.unsqueeze(raw_embedding, dim=0)
        attacked_embedding = torch.unsqueeze(attacked_embedding, dim=0)

        # cos_sim
        sim_metric = torch.nn.CosineSimilarity(dim=1)
        return sim_metric(raw_embedding, attacked_embedding)


    def extract_features(self, model, tokenizer, raw_input, attacked_input, timestep, max_length):

        """
        取出TextFooler攻击后的input的[CLS]

        inputs:
            train batch in each step from dataloader
        hidden_states:
            bert_base output.hidden_states
            shape: (layer_num, batch_size, sequence_length, hidden_size)
            layer_0: embedding, layer_1-12: bert encoder
        timestep:
            当前时间步
        max_length:
            不同数据集的max_length: imdb=512，sst2=32，ag_news=
        return:
            [CLS] of each sentence in 13 bert layers
            len(list) = batch_size
            [{'input_ids', 'label', 'layer_0', 'layer_1', 'layer_2', ...},
             {'input_ids', 'label', 'layer_0', 'layer_1', 'layer_2', ...},
             ... ]
        """


        # NOTE: attacked_input送入bert，用于获取连续的[cls]表示
        inputs = tokenizer(
            attacked_input,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(model.parameters()).device
        inputs.to(model_device)
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        # print(hidden_states)

        # NOTE: 这里保存raw_inputs经过tokenizer后的结果，而不是attacked_input送入bert，为了和freelb的结果进行对齐
        raw_inputs = tokenizer(
            raw_input,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        feature_dict = dict(raw_inputs)    # {input_ids: tensor(batch_size,512), token_type_ids tensor(batch_size,512), ...}
        # feature_dict['raw_input'] = raw_input
        # feature_dict['attacked_input'] = attacked_input
        # feature_dict['timestep'] = timestep
        # batch_size = 1
        batch_size = feature_dict['input_ids'].shape[0]

        for layer_id in range(13):
            layer_key = 'layer_' + str(layer_id)
            # hidden_states[layer_id][:] (batch_size, max_len, 768)
            feature_dict[layer_key] = hidden_states[layer_id][:][:, 0, :]    # CLS: (batch_size, 768)

        for i in range(batch_size):
            one_sentence = {k: v[i].detach().cpu() for k, v in feature_dict.items()
                            if k not in ['token_type_ids', 'attention_mask']}
            one_sentence['raw_input'] = raw_input
            one_sentence['attacked_input'] = attacked_input
            # NOTE: 12.5 此处添加了获取use_sim的代码
            one_sentence['use_sim'] = self.get_use_sim(raw_input, attacked_input)
            one_sentence['timestep'] = timestep
            self.all_sentences.append(one_sentence)

    def save_to_file(self):
        torch.save(self.all_sentences, self.save_path)
        print('**************\n save to: {} successfully \n**************\n'.format(self.save_path))

