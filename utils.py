import math
import re
from collections import Counter
from itertools import chain


def prepare_data(filename, min_count=5, minlen=5):
    with open(filename, encoding='utf-8') as texts:
        tag = re.compile(r'_.+')
        sentences = [
            [re.sub(tag, '', token_tag).lower() for token_tag in line.split()]
            for line in texts]
        sentences = [sent for sent in sentences if len(sent) >= minlen]

    words_freqs = Counter(chain.from_iterable(sentences)).most_common()
    words_freqs.append(('<unk>', min_count))
    words_freqs = [('<pad>', min_count)] + words_freqs
    words, frequencies = zip(*[x for x in words_freqs if x[1] >= min_count])
    word2id = {w: i for i, w in enumerate(words)}
    id2word = [key for key, val in word2id.items()]

    return sentences, frequencies, word2id, id2word


def texts_to_sequences(texts, word2id, oov_token='<unk>'):
    return [[word2id[token] if token in word2id else word2id[oov_token]
             for token in line] for line in texts]


def pad_sequences(sequences, maxlen, value=0):
    for i, seq in enumerate(sequences):
        sequences[i] += [value] * (maxlen - len(seq))
    return sequences


def make_sampling_table(word_freqs, sampling_factor=1e-5):
    return [1 - math.sqrt(sampling_factor / freq_i) for freq_i in word_freqs]


def labels_to_tsv(words):
    with open('labels.tsv', 'w+', encoding='utf-8') as labels_file:
        for word in words.keys():
            labels_file.write('{}\n'.format(word))
