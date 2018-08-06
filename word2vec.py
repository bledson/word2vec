import numpy as np
import random
import tensorflow as tf

data_filename = 'macmorpho-train.txt'
win_size = 5
neg_samples = 15
emb_dim = 50
sample_factor = 1e-5
smooth_factor = .75


def skipgrams(sequence,
              window_size=5,
              shuffle=True,
              sampling_table=None,
              seed=None):
    couples = []
    labels = []
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
                couples.append([wi, wj])
                labels.append(1)

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels


with tf.Graph().as_default() as graph:
    with tf.device('/cpu:0'):
        with tf.name_scope(''):
            pass
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    writer = tf.summary.FileWriter('graphs', graph=graph)
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.Print())
