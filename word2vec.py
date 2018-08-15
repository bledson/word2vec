import math
import numpy as np
import random
import re
import tensorflow as tf
from collections import Counter
from itertools import chain

data_filename = 'macmorpho-train.txt'
batch_sz = 128
eps = 201
min_count = 5
win_sz = 5
neg_samples = 15
emb_dim = 50
sampling_fac = 1e-5
distortion_fac = .75


def make_sampling_table(word_freqs, sampling_factor=1e-5):
    return [1 - math.sqrt(sampling_factor / freq_i) for freq_i in word_freqs]


def skipgrams(sequence,
              window_size=5,
              shuffle=True,
              sampling_table=None,
              seed=None):
    words = []
    contexts = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                words.append(wi)
                contexts.append(wj)

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(words)
        random.seed(seed)
        random.shuffle(contexts)

    return words, contexts


def tokenizer(filename):
    with open(filename, encoding='utf-8') as texts:
        num = re.compile(r'_.+')
        sentences = [[re.sub(num, '', token).lower() for token in line.split()]
                     for line in texts]

    words_freqs = Counter(chain.from_iterable(sentences)).most_common()
    words_freqs.append(('unk', min_count + 1))
    words, frequencies = zip(*[x for x in words_freqs if x[1] > min_count])
    word2id = {w: i for i, w in enumerate(words)}
    id2word = [key for key, val in word2id.items()]

    return sentences, frequencies, word2id, id2word


def texts_to_sequences(texts, word2id):
    return [[word2id[token] if token in word2id else word2id['unk']
             for token in line] for line in texts]


sents, freqs, w2id, id2w = tokenizer(data_filename)
sequences = texts_to_sequences(sents, w2id)
sampling_tbl = make_sampling_table(freqs, sampling_factor=sampling_fac)
wrds = []
ctxts = []
for seq in sequences:
    w_, c_ = skipgrams(seq, window_size=win_sz, sampling_table=sampling_tbl)
    wrds += w_
    ctxts += c_

with tf.Graph().as_default() as graph:
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int64, [batch_sz], name='inputs')
        labels = tf.placeholder(tf.int64, [batch_sz, 1], name='labels')
        sampled_ids = tf.placeholder(tf.int64, [neg_samples], name='samp_ids')
        true_count = tf.placeholder(tf.float64, [batch_sz, 1], name='true_cnt')
        sampled_count = tf.placeholder(tf.float64, [neg_samples], name='samp_cnt')

    with tf.name_scope('embeddings'):
        embeddings = tf.get_variable(
            'embeddings',
            shape=[len(w2id), emb_dim],
            dtype=tf.float64,
            initializer=tf.glorot_uniform_initializer())
        embs = tf.nn.embedding_lookup(embeddings, inputs)

    with tf.name_scope('weights'):
        loss_w = tf.get_variable(
            'loss_weights',
            shape=[len(w2id), emb_dim],
            dtype=tf.float64,
            initializer=tf.glorot_uniform_initializer())

    with tf.name_scope('biases'):
        loss_b = tf.get_variable(
            'loss_biases',
            shape=[len(w2id)],
            dtype=tf.float64,
            initializer=tf.ones_initializer)

    with tf.name_scope('loss'):
        loss = tf.nn.sampled_softmax_loss(
            weights=loss_w,
            biases=loss_b,
            labels=labels,
            inputs=embs,
            num_sampled=neg_samples,
            num_classes=len(w2id),
            sampled_values=(sampled_ids, true_count, sampled_count))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter('./graphs', session=sess)
        sess.run(tf.global_variables_initializer())

        average_loss = 0
        for ep in range(eps):
            for step in range(len(w2id) // batch_sz):
                inputs = wrds[batch_sz * step:batch_sz * (step + 1)]
                inputs = tf.convert_to_tensor(inputs, tf.int64)
                labels = ctxts[batch_sz * step:batch_sz * (step + 1)]
                labels_matrix = tf.reshape(tf.cast(labels, tf.int64),
                                           [batch_sz, 1])
                print('teste')
                sampled_ids, true_count, sampled_count = \
                    tf.nn.fixed_unigram_candidate_sampler(
                        true_classes=labels_matrix,
                        num_true=1,
                        num_sampled=neg_samples,
                        unique=True,
                        range_max=len(w2id),
                        distortion=distortion_fac,
                        unigrams=freqs)

                feed_dict = {inputs: inputs,
                             labels: labels_matrix,
                             sampled_ids: sampled_ids,
                             true_count: true_count,
                             sampled_count: sampled_count}

                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 100 == 0:
                    if step > 0:
                        average_loss /= 2000
                        print(step, ': ', average_loss)
                average_loss = 0

    writer.close()
