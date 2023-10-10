"""
Random Word Swap by Glove Embedding
-------------------------------

Based on paper: `<arxiv.org/abs/1603.00892>`_

Paper title: Counter-fitting Word Vectors to Linguistic Constraints

"""
import numpy as np

from textattack.shared import AbstractWordEmbedding, WordEmbedding

from .word_swap import WordSwap


class WordSwapRandomWordSubstitution1225(WordSwap):
    """Transforms an input by replacing its words with synonyms in the word
    embedding space.

    # NOTE 20221225 Random replace a word in a sentence or phrase. based on glove embedding.
    """

    def __init__(self, embedding=None, **kwargs):
        super().__init__(**kwargs)
        if embedding is None:
            embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
            )
        self.embedding = embedding
        print("**********" * 5)
        print("building WordSwapRandomWordSubstitution1225 successfully ...")
        print("**********" * 5)
        print()

    def _get_replacement_words(self, word):
        # NOTE 20221225 Random replace a word in a sentence or phrase. based on glove embedding.
        for _ in range(1000):
            sub_id = np.random.randint(0, 65713)    # 65713 * 300
            sub_word = self.embedding.index2word(sub_id)
            # 如果出现sub_word="aaaa--aa"的情况，那么分词时会出现错误，暂时跳过这种情况
            if "-" not in word and "-" in sub_word:
                continue
            # 如果出现word="love", sub_word="it's"的情况，那么分词时会出现错误，暂时跳过这种情况
            if "'" not in word and "'" in sub_word:
                continue
            candidate_words = []
            candidate_words.append(recover_word_case(sub_word, word))
            return candidate_words

    def extra_repr_keys(self):
        return ["embedding"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
