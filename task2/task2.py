import numpy as np
import nltk
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
from nltk.corpus import brown
# import brown
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec

from DocumentBasedVectorizer import DocumentBasedVectorizer
from ContextBasedVectorizer import ContextBasedVectorizer


def cluster_and_evaluate(window_size, target_words):
    evaluated_context_based_vectorizer = ContextBasedVectorizer(target_words=target_words,
                                                                corpus=brown,
                                                                tagged=True,
                                                                window_size=window_size,
                                                                stem=True,
                                                                lemmatize=True,
                                                                remove_stopwords=True,
                                                                for_evaluation=True)

    evaluated_context_based_vectors = evaluated_context_based_vectorizer.vectorize()

    clusterer = KMeansClusterer(
        50, euclidean_distance, avoid_empty_clusters=True)
    clusters = clusterer.cluster(
        list(evaluated_context_based_vectors.values()))

    print(clusters)
    appended = []
    for word in target_words:
        appended.append(word)

    for word in target_words:
        appended.append(word[::-1])

    vectors = list(evaluated_context_based_vectors.values())
    correct = 0
    result = []
    for i in range(0, len(target_words)):
        word_vector = vectors[i]
        reversed_word_vector = vectors[i + len(target_words)]
        result.append([{appended[i]: clusterer.classify(word_vector)}, {
                      appended[i + len(target_words)]: clusterer.classify(reversed_word_vector)}])
        if (clusterer.classify(word_vector) == clusterer.classify(reversed_word_vector)):
            correct += 1

    return (correct / len(target_words)) * 100


def cluster(target_words, corpus, number_of_clusters, distance):
    vectorizer = ContextBasedVectorizer(target_words=target_words,
                                        corpus=corpus,
                                        tagged=True,
                                        stem=True,
                                        lemmatize=True,
                                        remove_stopwords=True,
                                        for_evaluation=True)
    vectors = vectorizer.vectorize()

    clusterer = KMeansClusterer(number_of_clusters,
                                distance,
                                avoid_empty_clusters=True)
    clusters = clusterer.cluster(vectors.values(), assign_clusters=True)
    return clusters
    # cluster_dict = {}
    # for i in range(0, len(target_words) - 1):
    #     cluster = clusters[i]
    #     if (cluster in cluster_dict):
    #         cluster_dict[cluster].append(target_words[i])
    #     else:
    #         cluster_dict[cluster] = [target_words[i]]

    # return cluster_dict


# def accuracy(clusters):
#     for cluster in clusters:


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

    # print(cluster(nltk.pos_tag(target_words), brown, number_of_clusters=4,
    #               distance=euclidean_distance))

    cluster_and_evaluate(10, nltk.pos_tag(target_words))
    # output = {}
    # for window_size in range(2, 10):
    #     output[window_size] = str(cluster_and_evaluate(
    #         window_size, nltk.pos_tag(target_words))) + '%'

    # print(output)
