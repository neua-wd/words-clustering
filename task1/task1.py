import nltk
from nltk.corpus import brown

tagged_words = brown.tagged_words()
# tagged_words = [('a', 'NN'), ('b', 'VB'), ('c', 'NN'), ('d', 'AB')]


def word_likelihood(word, tag):
    word_with_tag_count = 0
    tag_count = 0

    for tagged_word in tagged_words:
        if tagged_word[1] == tag:
            tag_count += 1
            if tagged_word[0].lower() == word.lower():
                word_with_tag_count += 1

    return (word_with_tag_count) / (tag_count)


def tag_transition_probability(tag, preceding_tag):
    tag_follows_count = 0
    preceding_tag_count = 0

    for i in range(1, len(tagged_words)):
        if tagged_words[i - 1][1] == preceding_tag:
            preceding_tag_count += 1
            if tagged_words[i][1] == tag:
                tag_follows_count += 1

    return tag_follows_count / preceding_tag_count


print(word_likelihood('race', 'NN'))
print(tag_transition_probability('NN', 'VB'))
