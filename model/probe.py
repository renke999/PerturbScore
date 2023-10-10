import torch


class FeatureProbe:
    """
    feature probe for bert_base(12 layer)
    a util class just for extracting [CLS] token embedding
    """

    def __init__(self, save_path='freelb_tensor.pt'):
        """
        model: bert-base-uncased finetuned on imdb
        save_path: torch.save path
        """
        self.all_sentences = []
        self.save_path = save_path

    def extract_features(self, model, inputs, input_ids, timestep=0):
        """
        inputs:
            train batch in each step from dataloader
        hidden_states:
            bert_base output.hidden_states
            shape: (layer_num, batch_size, sequence_length, hidden_size)
            layer_0: embedding, layer_1-12: bert encoder
        return:
            [CLS] of each sentence in 13 bert layers
            len(list) = batch_size
            [{'input_ids', 'label', 'layer_0', 'layer_1', 'layer_2', ...},
             {'input_ids', 'label', 'layer_0', 'layer_1', 'layer_2', ...},
             ... ]
        """
        model.eval()
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        # print(hidden_states)

        feature_dict = dict(inputs)    # {input_ids: tensor(batch_size,512), token_type_ids tensor(batch_size,512), ...}
        feature_dict['input_ids'] = input_ids
        batch_size = feature_dict['input_ids'].shape[0]


        for layer_id in range(13):
            layer_key = 'layer_' + str(layer_id)
            # hidden_states[layer_id][:] (batch_size, max_len, 768)
            feature_dict[layer_key] = hidden_states[layer_id][:][:, 0, :]    # CLS: (batch_size, 768)

        for i in range(batch_size):
            one_sentence = {k: v[i].detach().cpu() for k, v in feature_dict.items()
                            if k not in ['token_type_ids', 'attention_mask']}
            one_sentence['timestep'] = timestep
            self.all_sentences.append(one_sentence)

    def save_to_file(self):
        torch.save(self.all_sentences, self.save_path)
