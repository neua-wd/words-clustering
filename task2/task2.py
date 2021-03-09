import numpy as np
import nltk
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
from nltk.corpus import brown
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec

from DocumentBasedVectorizer import DocumentBasedVectorizer


def cluster(target_words, corpus, number_of_clusters, distance):
    vectorizer = DocumentBasedVectorizer(
        target_words, corpus, stem=True, lemmatize=True, remove_stopwords=True)
    vectors = vectorizer.vectorize(weighting='tfidf')

    clusterer = KMeansClusterer(
        number_of_clusters, distance, avoid_empty_clusters=True)
    clusters = clusterer.cluster(vectors.values(), assign_clusters=True)

    cluster_dict = {}
    for i in range(0, len(target_words) - 1):
        if (clusters[i] in cluster_dict):
            cluster_dict[clusters[i]].append(target_words[i])
        else:
            cluster_dict[clusters[i]] = []

    return cluster_dict


def nltk_cluster(target_words, corpus, number_of_clusters):
    model = Word2Vec(corpus.sents(), min_count=5, workers=6, window=10)
    X = model[model.wv.vocab]

    vector_model = Word2Vec(target_words, min_count=5, workers=6, window=10)
    vectors = vector_model[vector_model.wv.vocab]

    clusterer = KMeansClusterer(number_of_clusters, euclidean_distance)
    clusterer.cluster(X, assign_clusters=True)

    classified = []
    for vector in vectors:
        classified.append(clusterer.classify(vector))

    print(classified)


def sklearn_cluster(target_words, corpus, number_of_clusters):
    model = Word2Vec(corpus.sents(), min_count=1, workers=6)
    X = model[model.wv.vocab]

    vector_model = Word2Vec(target_words, min_count=1)
    vectors = vector_model[vector_model.wv.vocab]

    clusterer = KMeans(number_of_clusters)
    clusterer.fit(X)

    print(clusterer.predict(vectors))


if __name__ == "__main__":
    target_words = ['abstraction', 'actually', 'add', 'address', 'answer',
                    'argument', 'arguments', 'back', 'call', 'car', 'case',
                    'cdr', 'computer', 'course', 'dictionary', 'different',
                    'evaluator', 'function', 'general', 'got', 'idea', 'kind',
                    'lambda', 'machine', 'mean', 'object', 'operator', 'order',
                    'pair', 'part', 'particular', 'pattern', 'place', 'problem',
                    'process', 'product', 'program', 'reason', 'register',
                    'result', 'set', 'simple', 'structure', 'system', 'they',
                    'together', 'using', 'variable', 'why', 'zero']

    # sklearn_cluster(target_words, brown, 50, 'count')
    # start = time.perf_counter()
    # nltk_cluster(target_words, brown, 50)
    # sklearn_cluster(target_words, brown, 10)
    # finish = time.perf_counter()
    # print('finished in: ' + str(round(finish - start, 2)) + 'seconds')

    print(cluster(target_words, brown, number_of_clusters=5,
                  distance=cosine_distance))

    # vectorizer = DocumentBasedVectorizer(
    #     target_words, brown, stem=True, lemmatize=True, remove_stopwords=True)
    # print(vectorizer.vectorize(weighting='tfidf'))
