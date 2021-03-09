from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import word_tokenize
import numpy as np
from math import log


# Vectorizer class, using binary weights
# Takes an array of target words, the corpus to be learned, and boolean values
# indicating whether or not to stem, lemmatize and/or remove stopwords
class DocumentBasedVectorizer:
    def __init__(self, target_words, corpus, stem=False, lemmatize=False,
                 remove_stopwords=False):
        if (stem or lemmatize):
            self.target_words = self.__normalize(target_words)
        else:
            self.target_words = target_words

        self.documents = self.__get_documents(
            corpus, stem, lemmatize, remove_stopwords)

    def vectorize(self, weighting='binary'):
        if (weighting == 'binary'):
            return self.__binary_matrix()
        elif (weighting == 'term-frequency'):
            return self.__tf_matrix()
        elif (weighting == 'tfidf'):
            return self.__tfidf_matrix()

    # Returns a term-document frequency as a { word: word_vector } dictionary,
    # where each word vector vector is a numpy array
    def __binary_matrix(self):
        matrix = {}
        for word in self.target_words:
            word_vector = []
            for document in self.documents:
                if (word in document):
                    word_vector.append(1)
                else:
                    word_vector.append(0)

            matrix[word] = np.array(word_vector)

        return matrix

    def __tf_matrix(self):
        matrix = {}
        for word in self.target_words:
            word_vector = []
            for document in self.documents:
                word_vector.append(document.count(word))

            matrix[word] = np.array(word_vector)

        return matrix

    def __tfidf_matrix(self):
        tf_matrix = self.__tf_matrix()
        M = len(self.documents)

        matrix = {}
        for word in self.target_words:
            word_vector = []
            for i in range(0, len(self.documents)):
                tf = tf_matrix[word][i]
                count = np.count_nonzero(tf_matrix[word])
                mk = count if count > 0 else M
                idf = log(M / mk)
                word_vector.append(tf * idf)

            matrix[word] = np.array(word_vector)

        return matrix

    def __get_documents(self, corpus, stem, lemmatize, remove_stopwords):
        documents = []
        for category in corpus.categories():
            document = corpus.words(categories=category)
            if (remove_stopwords):
                document = [w for w in document if w.lower(
                ) not in stopwords.words('english')]

            if (stem | lemmatize):
                document = self.__normalize(document)

            documents.append(document)

        return documents

    def __normalize(self, words):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        normalized_words = []
        for word in words:
            if (lemmatizer):
                word = lemmatizer.lemmatize(word)
            if (stemmer):
                word = stemmer.stem(word)

            normalized_words.append(word)

        return normalized_words
