"""

TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)

"""

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding
from textattack.transformations import WordSwapRandomWordSubstitution1225


from .attack_recipe import AttackRecipe


class RandomSwap1225(AttackRecipe):
    """
    random swap a word
    """

    @staticmethod
    def build(model_wrapper, save_path, dataset):

        print()
        print("**********" * 5)
        print("building RandomSwap1225 successfully ...")
        print("**********" * 5)
        print()

        #
        # Random swap words with embeddings
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapRandomWordSubstitution1225()
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)


        # 保证代码的完整性，其实没什么用
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        # constraints.append(use_constraint)


        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)

        #
        # Randomly swap words with "Word Importance Ranking".
        #
        # NOTE model wrapper, use_encoder都是自行添加的，避免显存爆炸
        search_method = GreedyWordSwapWIR(model_wrapper, use_constraint, save_path, dataset, wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)
