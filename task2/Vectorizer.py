from nltk.corpus import stopwords
from nltk import pos_tag, PorterStemmer, WordNetLemmatizer, word_tokenize
import numpy as np
from math import log, floor
from random import sample


# Vectorizer class, using binary weights
# Takes an array of target words, the corpus to be learned, and boolean values
# indicating whether or not to stem, lemmatize and/or remove stopwords
class Vectorizer:
    def __init__(self, target_words, corpus, tagged=False, stem=False,
                 lemmatize=False, remove_stopwords=False, for_evaluation=False):
        if (for_evaluation):
            self.target_words = self.__append_reversed(target_words)
        else:
            self.target_words = target_words

        self.corpus = corpus
        self.tagged = tagged
        self.stem = stem
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.for_evaluation = for_evaluation

    def __get_documents(self):
        documents = []
        for file_id in self.corpus.fileids():
            if (self.tagged):
                document = self.__tagged_words(file_id)
            else:
                document = self.corpus.words(fileids=file_id)

            if (self.remove_stopwords):
                if (self.tagged):
                    document = [w for w in document if w[0].lower(
                    ) not in stopwords.words('english')]
                else:
                    document = [w for w in document if w.lower(
                    ) not in stopwords.words('english')]

            if (self.stem or self.lemmatize):
                document = self.__normalize(document)

            if (self.for_evaluation):
                document = self.__randomly_reverse_half(
                    list(document), self.target_words)

            documents.append(document)

        return documents

    def __tagged_words(self, file_id):
        if ('nltk.corpus.reader.tagged' in str(self.corpus.__class__)):
            return self.corpus.tagged_words(fileids=file_id)[0:500]
        else:
            return pos_tag(self.corpus.words(fileids=file_id))[0:500]
    # Takes an array of words
    # Returns the normalized (stemmed and/or lemmatized) version
    # of those words

    def __normalize(self, words):
        if (self.stem):
            stemmer = PorterStemmer()
        if (self.lemmatize):
            lemmatizer = WordNetLemmatizer()

        normalized_words = []
        for word in words:
            if (self.lemmatize):
                if (self.tagged):
                    word = (lemmatizer.lemmatize(word[0]), word[1])
                else:
                    word = lemmatizer.lemmatize(word)
            if (self.stem):
                if (self.tagged):
                    word = (stemmer.stem(word[0]), word[1])
                else:
                    word = stemmer.stem(word)

            normalized_words.append(word)

        return normalized_words

    # Takes an array of words and an array of target words
    # For each word in the target words, randomly reverse half of its
    # occurrences in the 'words' array
    # Returns the result
    def __randomly_reverse_half(self, words, target_words):
        for word in target_words[0:(int(len(target_words) / 2))]:
            occur_at = list(np.where(np.array(words) == word)[0])
            indexes_to_reverse = sample(occur_at, floor(len(occur_at) / 2))

            for index in indexes_to_reverse:
                words[index] = self.__reverse(words[index])

        return words

    # Takes an array of words
    # Returns the array appended by the reverse of each word in the array
    def __append_reversed(self, words):
        appended = []
        for word in words:
            appended.append(word)

        for word in words:
            appended.append(self.__reverse(word))

        return appended

    def __reverse(self, string):
        return string[::-1]
