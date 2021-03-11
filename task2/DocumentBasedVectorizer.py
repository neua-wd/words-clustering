from Vectorizer import Vectorizer
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import word_tokenize
import numpy as np
from math import log


# Vectorizer class, using binary weights
# Takes an array of target words, the corpus to be learned, and boolean values
# indicating whether or not to stem, lemmatize and/or remove stopwords
class DocumentBasedVectorizer(Vectorizer):
    def __init__(self, corpus, target_words, weighting, tagged=False, stem=False,
                 lemmatize=False, remove_stopwords=False, for_evaluation=False):
        super().__init__(target_words, corpus, tagged, stem, lemmatize,
                         remove_stopwords, for_evaluation)

        self.weighting = weighting
        self.documents = self._Vectorizer__get_documents()

    def vectorize(self):
        if (self.stem or self.lemmatize):
            self._Vectorizer__normalize(self.target_words)

        if (self.weighting == 'binary'):
            return self.__binary_matrix(self.target_words)
        elif (self.weighting == 'term-frequency'):
            return self.__tf_matrix(self.target_words)
        elif (self.weighting == 'tfidf'):
            return self.__tfidf_matrix(self.target_words)

    # Returns a term-document frequency as a { word: word_vector } dictionary,
    # where each word vector vector is a numpy array
    def __binary_matrix(self, target_words):
        matrix = {}
        for word in target_words:
            word_vector = []
            for document in self.documents:
                if (word in document):
                    word_vector.append(1)
                else:
                    word_vector.append(0)

            matrix[word] = np.array(word_vector)

        return matrix

    def __tf_matrix(self, target_words):
        matrix = {}
        for word in target_words:
            word_vector = []
            for document in self.documents:
                word_vector.append(document.count(word))

            matrix[word] = np.array(word_vector)

        return matrix

    def __tfidf_matrix(self, target_words):
        tf_matrix = self.__tf_matrix(target_words)
        M = len(self.documents)

        matrix = {}
        for word in target_words:
            word_vector = []
            for i in range(0, len(self.documents)):
                tf = tf_matrix[word][i]
                count = np.count_nonzero(tf_matrix[word])
                mk = count if count > 0 else M
                idf = log(M / mk)
                word_vector.append(tf * idf)

            matrix[word] = np.array(word_vector)

        return matrix
