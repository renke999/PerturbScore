"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


# NOTE ========================== code for probe ===========================
from textattack.probe import FeatureProbe

# bert-base-uncased
# feature_probe = FeatureProbe(save_path='/remote-home/kren/exps/TextAttack/bert_textattack_tensor.pt')

# 12.1 攻击到时间步t后不再继续攻击，模型使用freelb
# save_path = '/remote-home/kren/exps/TextAttack/textattack_freelb_raw_tensor.pt'

# 12.5 攻击到时间步t后继续攻击，加入USE_encoder相似度，模型使用freelb
# save_path = '/remote-home/kren/exps/TextAttack/textattack_freelb_use_tensor.pt'

# 12.7 测试更多的不同的t，查看cos(f(x), f(x'))，获取更细粒度的变化
# save_path = '/remote-home/kren/exps/TextAttack/textattack_freelb_use_tensor_imdb_20221207.pt'

# 12.11 a. 测试imdb 1000个sample，10个t，freelb
# save_path = '/remote-home/kren/exps/TextAttack/textattack_freelb_imdb_1000sample_20221211.pt'

# 12.11 b. 测试imdb 1000个sample，10个t，bert
# save_path = '/remote-home/kren/exps/TextAttack/textattack_bert_imdb_1000sample_20221211.pt'

# 12.11 c. 测试sst2 1000个sample，t短，freelb
# save_path = '/remote-home/kren/exps/TextAttack/textattack_freelb_sst2_1000sample_20221211.pt'

# 12.11 d. 测试sst2 1000个sample，t短，bert
# save_path = '/remote-home/kren/exps/TextAttack/textattack_bert_sst2_1000sample_20221211.pt'

# 12.26 a. 测试imdb-freelb-random
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_imdb_freelb_random_1000sample_20221226.pt'
# 12.26 b. 测试imdb-bert-random
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_imdb_bert_random_1000sample_20221226.pt'

# 12.28 a. 测试agnews_freelb_random
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_freelb_random_1000sample_20221230.pt'

# 12.28 b. 测试agnews-bert-random
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_bert_random_1000sample_20221230.pt'

# 12.28 c. 测试agnews-freelb-textfooler
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_freelb_textfooler_1000sample_20221230.pt'

# 12.28 d. 测试agnews-bert-textfooler
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_bert_textfooler_1000sample_20221230.pt'

# 1.12 做一下table1的数据
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_bert_textfooler_100sample_20220112.pt'

# 1.18 做一下不同扰动的数据
# save_path = '/remote-home/kren/exps/TextAttack/tensors/textfooler_agnews_bert_rand_100sample_20220118.pt'


# we initialize save_path and dateset using CLI
feature_probe = FeatureProbe()
# NOTE ====================================== End for Probe =============================




class GreedyWordSwapWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, model_wrapper, use_encoder, save_path, dataset, wir_method="unk", unk_token="[UNK]"):
        # NOTE ========================== code for probe ===========================
        # add model wrapper and use_encoder to extract feature
        # 加use_encoder是因为这是基于tensorflow的，如果处理不当会占用显存
        self.model_wrapper = model_wrapper
        feature_probe.set_use_encoder(use_encoder)
        feature_probe.set_save_path(save_path)
        feature_probe.set_dataset(dataset)
        # NOTE ====================================== End for Probe =============================
        self.wir_method = wir_method
        self.unk_token = unk_token

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        # self.search_method.get_indices_to_order = self.get_indices_to_order = word_transformation.__call__()
        # attack.py: get_indices_to_order()
        # return_indices (bool = True): Whether the function returns indices_to_modify instead of the transformed_texts.
        # Applies ``pre_transformation_constraints`` to ``text`` to get all the indices that can be used to search and order.
        # (textfooler中没有pre_transformation_constraints, indices_to_order默认是所有index)

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in indices_to_order:

                # Exit Loop when search_over is True - but we need to make sure delta_ps
                # is the same size as softmax_saliency_scores
                if search_over:
                    delta_ps = delta_ps + [0.0] * (
                        len(softmax_saliency_scores) - len(delta_ps)
                    )
                    break

                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(
                    transformed_text_candidates
                )
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            print()
            print("**********" * 5)
            print("using delete sort...")
            print("**********" * 5)
            print()
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len_text)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, index in enumerate(indices_to_order):
                matched_tokens = word2token_mapping[index]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        # NOTE 1225，采用random换词的策略（应该会使两个句子相似度更差？）
        elif self.wir_method == "random":
            print()
            print("**********" * 5)
            print("using random sort...")
            print("**********" * 5)
            print()
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        # 如果不是random，则按照index_scores进行排序（textfooler是采用delete的策略）
        if self.wir_method != "random":
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over

    def perform_search(self, initial_result):
        """
        搜索方法 SearchMethod 以初始的 GoalFunctionResult 作为输入，返回最终的 GoalFunctionResult。
        get_transformations 方法，以一个 AttackedText 对象作为输入，返还所有符合约束条件的变换结果。
        搜索方法不断地调用 get_transformations 函数，直到攻击成功 (由 get_goal_results 决定) 或搜索结束。
        """
        # NOTE Perturbs attacked_text from initial_result until goal is reached or search is exhausted.

        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        # NOTE 1225 使用random策略
        """
        Returns word indices of ``initial_text`` in descending order of importance.
        """
        index_order, search_over = self._get_index_order(attacked_text)
        i = 0
        cur_result = initial_result
        results = None

        # NOTE ========================== code for probe: adv_step = 0 ===========================
        adv_step = 0
        raw_text = attacked_text.tokenizer_input
        changed_text = cur_result.attacked_text.tokenizer_input
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        max_length = self.model_wrapper.max_length
        # timestep=0 原始的输入情况
        print("probe: " + str(adv_step))
        feature_probe.extract_features(model, tokenizer, raw_text, changed_text, adv_step, max_length)
        # NOTE ====================================== End for Probe =============================


        """
        根据index_order顺序进行攻击，i是也就是攻击步数
        """

        while i < len(index_order) and not search_over:
            # len(index_order) 一般在[70, 150]区间
            # print(len(index_order))
            # return cur_result
            """
            Args:
                current_text: The current ``AttackedText`` on which to perform the transformations.
                original_text: The original ``AttackedText`` from which the attack started.
            Returns:
                A filtered list of transformations where each transformation matches the constraints
                
            attack.py中定义   
            self.search_method.get_transformations = self.get_transformations = WordSwapEmbedding
            
            WordSwapEmbedding
            transformation: word_embedding(max=50) (all possible replacements of the selected word w_i)
            """

            # NOTE: 得到attacked_text集合
            # NOTE: 1225，改变get_transformations的策略（random选一个词替换）
            # indices_to_modify: 修改的index
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            # print('===============================test ')
            # print(transformed_text_candidates)

            """
            transformed_text_candidates: list，包含了加了constraint后所有可能的替换后的句子
            """
            i += 1
            adv_step += 1

            # NOTE 1226 random换词时，如果换的这个词含有"'"或者"-"，则跳过这个词，换下一个
            if self.wir_method == "random":
                modified_word = cur_result.attacked_text.words[index_order[i-1]]
                # print("modified word: {}".format(modified_word))
                if "'" in modified_word or "-" in modified_word:
                    # print("***************** pass *****************")
                    adv_step -= 1
                    continue

            if len(transformed_text_candidates) == 0:
                # 没有可替换的词，攻击失败，random攻击不会出现这个错误
                adv_step -= 1
                continue

            """
            在textattack/goal_functions/goal_function.py定义
            results是textattack/goal_functions/goal_function_results/classification_goal_function_result.py
            
            self.num_queries += len(attacked_text_list)
            
            self.search_method.get_goal_results = self.goal_function.get_results
            """

            # score = 1 - logits[self.ground_truth_output]，越大表示扰动的效果越好，随后按score降序排列
            # search_over: self.num_queries == self.query_budget, query_budget=inf（未设置查询上限）
            # NOTE: 1225，results应该长度为1，因为改变了get_transformations的策略（random选一个词替换）
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # print("len results: {}, if random, should be 1.".format(len(results)))

            # NOTE: 1225 取消Skip swaps which don't improve the score这个限制条件，直接cur_result = results[0]
            if self.wir_method == "random":
                # if results[0].score > cur_result.score:
                cur_result = results[0]
                # NOTE ========================== code for probe ===========================
                raw_text = attacked_text.tokenizer_input
                changed_text = cur_result.attacked_text.tokenizer_input
                model = self.model_wrapper.model
                tokenizer = self.model_wrapper.tokenizer
                max_length = self.model_wrapper.max_length

                # NOTE 20221225.a 选1000个imdb sample，选10个具有代表性的t进行尝试
                if feature_probe.get_dataset() == 'imdb':
                    if adv_step in [1, 3, 5, 8, 10, 13, 15, 20, 25, 30]:
                        print("probe: " + str(adv_step))
                        feature_probe.extract_features(model, tokenizer, raw_text, changed_text, adv_step, max_length)

                # NOTE 20221230 选1000个agnews sample，agnews数据集短，选较短的t进行尝试
                elif feature_probe.get_dataset() == 'ag_news':
                    if adv_step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]:
                        print("probe: " + str(adv_step))
                        feature_probe.extract_features(model, tokenizer, raw_text, changed_text, adv_step, max_length)


                # NOTE ====================================== End for Probe =============================

                # 20230112
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    feature_probe.save_to_file()
                    return cur_result

                # TODO 20221225 全部都攻击30步，30步后直接停止攻击，节省时间
                if adv_step <= 30:
                    continue
                break

            elif self.wir_method == "delete":
                # 如果满足results[0].score > cur_result.score，说明是一次有效的攻击
                if results[0].score > cur_result.score:
                    cur_result = results[0]
                    # NOTE ========================== code for probe ===========================
                    raw_text = attacked_text.tokenizer_input
                    changed_text = cur_result.attacked_text.tokenizer_input
                    model = self.model_wrapper.model
                    tokenizer = self.model_wrapper.tokenizer

                    # NOTE 20221225.a 选1000个imdb sample，选10个具有代表性的t进行尝试
                    if feature_probe.get_dataset() == 'imdb':
                        if adv_step in [1, 3, 5, 8, 10, 13, 15, 20, 25, 30]:
                            print("probe: " + str(adv_step))
                            feature_probe.extract_features(model, tokenizer, raw_text, changed_text, adv_step, max_length)

                    # NOTE 20221230 选1000个agnews sample，agnews数据集短，选较短的t进行尝试
                    elif feature_probe.get_dataset() == 'ag_news':
                        if adv_step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]:
                            print("probe: " + str(adv_step))
                            feature_probe.extract_features(model, tokenizer, raw_text, changed_text, adv_step, max_length)

                    # NOTE ====================================== End for Probe =============================


                    # 20230112
                    if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        feature_probe.save_to_file()
                        return cur_result

                    # TODO 20221225 全部都攻击30步，30步后直接停止攻击，节省时间
                    if adv_step <= 30:
                        continue
                    break
                # 如果不能满足results[0].score > cur_result.score，则代表此次攻击失败了
                else:
                    adv_step -= 1
                    continue

            else:
                print("wrong!!!!")
                break


            """
                class GoalFunctionResultStatus:
                SUCCEEDED = 0
                SEARCHING = 1  # In process of searching for a success
                MAXIMIZING = 2
                SKIPPED = 3
            """
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    # 由于已经降序排序好了，所以没有succeed时直接break就行
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        # NOTE 存储probe的结果
        feature_probe.save_to_file()

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
