import random

import numpy as np
import tensorflow as tf

from utils import *

DATA_FILENAME = 'macmorpho-train.txt'
BATCH_SIZE = 128
EPOCHS = 10
STEP = 40000
LEARNING_RATE = .25
MIN_COUNT = 6
MIN_LENGTH = 6
WINDOW_SIZE = 2
NUM_SAMPLED = 128
EMBEDDING_SIZE = 300
SAMPLING_FACTOR = 1e-5
SMOOTH_FACTOR = .75


def cbows(
        sequence,
        window_size=5,
        shuffle=True,
        sampling_table=None,
        seed=None):
    contexts = []
    words = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        contexts_wi = []
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                contexts_wi.append(wj)
        contexts.append(contexts_wi)
        words.append(wi)

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(words)
        random.seed(seed)
        random.shuffle(contexts)

    return contexts, words


def context_reduce_mean(contexts):
    mask = tf.not_equal(contexts, 0)
    valid_ids = tf.boolean_mask(contexts, mask)
    embs = tf.nn.embedding_lookup(embeddings, valid_ids)
    return tf.reduce_mean(embs, axis=0)


sents, freqs, w2id, id2w = prepare_data(
    DATA_FILENAME,
    min_count=MIN_COUNT,
    minlen=MIN_LENGTH)
labels_to_tsv(w2id)
sequences = texts_to_sequences(sents, w2id)
sampling_table = make_sampling_table(freqs, sampling_factor=SAMPLING_FACTOR)

contexts = []
words = []
for seq in sequences:
    c, w = cbows(
        seq,
        window_size=WINDOW_SIZE,
        sampling_table=sampling_table)
    contexts += c
    words += w
contexts = pad_sequences(contexts, 2 * WINDOW_SIZE)

with tf.Graph().as_default() as graph:
    with tf.name_scope('data'):
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(contexts, np.int64),
             tf.convert_to_tensor(words, np.int64)))
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
        embs = tf.map_fn(context_reduce_mean, inputs, dtype=tf.float32)

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
            initializer=tf.zeros_initializer(),
            trainable=False)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
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
                    num_reserved_ids=1,
                    unigrams=freqs[1:])))

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
        saver.save(
            sess,
            './models/skip-gram_{}d_{}n_{}w_{}c'.format(
                EMBEDDING_SIZE,
                NUM_SAMPLED,
                WINDOW_SIZE,
                MIN_COUNT),
            global_step)

        step = 0
        while True:
            step += 1
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                average_loss += loss_batch
                if step % STEP == 0:
                    saver.save(
                        sess,
                        './models/cbow_{}d_{}n_{}w_{}c'.format(
                            EMBEDDING_SIZE,
                            NUM_SAMPLED,
                            WINDOW_SIZE,
                            MIN_COUNT),
                        global_step)
                    print('Average loss at step {}: {:5.4f}'.format(
                        step, average_loss / STEP))
                    average_loss = 0.
            except tf.errors.OutOfRangeError:
                saver.save(
                    sess,
                    './models/cbow_{}d_{}n_{}w_{}c'.format(
                        EMBEDDING_SIZE,
                        NUM_SAMPLED,
                        WINDOW_SIZE,
                        MIN_COUNT),
                    global_step)
                break

        writer.close()
