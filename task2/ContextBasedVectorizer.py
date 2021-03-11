from Vectorizer import Vectorizer
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from numba import jit


class ContextBasedVectorizer(Vectorizer):
    def __init__(self, target_words, corpus, window_size, tagged=False,
                 stem=False, lemmatize=False, remove_stopwords=False,
                 for_evaluation=False):
        super().__init__(target_words, corpus, tagged, stem,
                         lemmatize, remove_stopwords, for_evaluation)
        self.contexts = self.__get_contexts(window_size)

    @jit(forceobj=True)
    def vectorize(self):
        if (self.stem or self.lemmatize):
            self._Vectorizer__normalize(self.target_words)

        matrix = {}
        for word_x in self.target_words:
            word_vector = []
            for word_y in self.target_words:
                co_occurences = 0
                for context in self.contexts:
                    if word_x != word_y and word_x in context and word_y in context:
                        co_occurences += 1

                word_vector.append(co_occurences)

            matrix[word_x] = np.array(word_vector)

        return matrix

    def __get_contexts(self, window_size):
        if (window_size == 'sentence'):
            if (self.tagged):
                corpus_words = list(self.corpus.tagged_words())
            else:
                corpus_words = list(self.corpus.words())

            return self.__get_sentences(corpus_words)
        elif (window_size == 'document'):
            return self._Vectorizer__get_documents()
        else:
            if (self.tagged):
                words = self.corpus.tagged_words()
            else:
                words = self.corpus.words()

            return [words[i:i + window_size] for i in range(
                    len(words) - window_size + 1)]

    def __get_sentences(self, corpus_words):
        if (self.for_evaluation):
            words = self._Vectorizer__randomly_reverse_half(
                corpus_words, self.target_words)
            return sent_tokenize(TreebankWordDetokenizer().detokenize(words))
        else:
            return self.corpus.sents()
