import math
import random
import re
from collections import Counter
from itertools import chain

import numpy as np
import tensorflow as tf

DATA_FILENAME = 'macmorpho-train.txt'
BATCH_SIZE = 128
EPOCHS = 100
STEP = 40000
LEARNING_RATE = .25
MIN_COUNT = 5
MIN_LENGTH = 5
WINDOW_SIZE = 2
NUM_SAMPLED = 128
EMBEDDING_SIZE = 300
SAMPLING_FACTOR = 1e-5
SMOOTH_FACTOR = .75


def skipgrams(
        sequence,
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


def prepare_data(filename):
    with open(filename, encoding='utf-8') as texts:
        num = re.compile(r'_.+')
        sentences = [[re.sub(num, '', token).lower() for token in line.split()]
                     for line in texts]
        sentences = [sent for sent in sentences if len(sent) > MIN_LENGTH]

    words_freqs = Counter(chain.from_iterable(sentences)).most_common()
    words_freqs.append(('<unk>', MIN_COUNT + 1))
    words, frequencies = zip(*[x for x in words_freqs if x[1] > MIN_COUNT])
    word2id = {w: i for i, w in enumerate(words)}
    id2word = [key for key, val in word2id.items()]

    return sentences, frequencies, word2id, id2word


def texts_to_sequences(texts, word2id):
    return [[word2id[token] if token in word2id else word2id['<unk>']
             for token in line] for line in texts]


def make_sampling_table(word_freqs, sampling_factor=1e-5):
    return [1 - math.sqrt(sampling_factor / freq_i) for freq_i in word_freqs]


def save_to_tsv(words):
    with open('labels.tsv', 'w+', encoding='utf-8') as labels_file:
        for word in words.keys():
            labels_file.write('{}\n'.format(word))


sents, freqs, w2id, id2w = prepare_data(DATA_FILENAME)
save_to_tsv(w2id)
sequences = texts_to_sequences(sents, w2id)
sampling_table = make_sampling_table(freqs, sampling_factor=SAMPLING_FACTOR)

words = []
contexts = []
for seq in sequences:
    w, c = skipgrams(
        seq,
        window_size=WINDOW_SIZE,
        sampling_table=sampling_table)
    words += w
    contexts += c

with tf.Graph().as_default() as graph:
    with tf.name_scope('data'):
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.asarray(words, np.int64),
             np.asarray(contexts, np.int64)))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
        iterator = dataset.make_initializable_iterator()
        inputs, labels = iterator.get_next()
        labels_matrix = tf.reshape(labels, [BATCH_SIZE, 1])

    with tf.name_scope('embeddings'):
        embeddings = tf.get_variable(
            'embeddings',
            shape=[len(w2id), EMBEDDING_SIZE],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer())
        tf.assign(embeddings[0], tf.zeros(EMBEDDING_SIZE))
        embs = tf.nn.embedding_lookup(embeddings, inputs)

    with tf.name_scope('weights'):
        sm_w = tf.get_variable(
            'sm_weights',
            shape=[len(w2id), EMBEDDING_SIZE],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer())

    with tf.name_scope('biases'):
        sm_b = tf.get_variable(
            'sm_biases',
            shape=[len(w2id)],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=sm_w,
            biases=sm_b,
            labels=labels_matrix,
            inputs=embs,
            num_sampled=NUM_SAMPLED,
            num_classes=len(w2id),
            sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=NUM_SAMPLED,
                unique=True,
                range_max=len(w2id),
                distortion=SMOOTH_FACTOR,
                unigrams=freqs)))

    with tf.name_scope('optimizer'):
        global_step = tf.Variable(
            0,
            dtype=tf.int32,
            name='global_step',
            trainable=False)
        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, .9).minimize(
            loss,
            global_step=global_step)

    saver = tf.train.Saver(name='saver', max_to_keep=1)
    with tf.Session(graph=graph) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())

        average_loss = 0.
        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        step = 0
        while True:
            step += 1
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                average_loss += loss_batch
                if step % STEP == 0:
                    saver.save(
                        sess,
                        './models/skip-gram_{}d_{}n'.format(EMBEDDING_SIZE,
                                                            NUM_SAMPLED),
                        global_step)
                    print('Average loss at step {}: {:5.4f}'.format(
                        step, average_loss / STEP))
                    average_loss = 0.
            except tf.errors.OutOfRangeError:
                saver.save(
                    sess,
                    './models/skip-gram_{}d_{}n'.format(EMBEDDING_SIZE,
                                                        NUM_SAMPLED),
                    global_step)
                break

        writer.close()
