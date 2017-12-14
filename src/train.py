import glob
import itertools
from os import path

from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from data.tfrecords_read import parse_example
from data.category_counts import category_counts
from model import model
from model.loss_function import *
from utils import data_processed_dir

filenames = glob.glob(path.join('../data/processed', '*.tfrecord'))
dataset = (
    tf.data.TFRecordDataset(filenames, compression_type='ZLIB')
        .map(parse_example)
        .batch(10)
)
dataset_iterator = dataset.make_initializable_iterator()
next_op = dataset_iterator.get_next()

logits = model(next_op['features'])
# TODO make it loss with negative sampling
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_op['cat_id'], logits=logits))
loss = negative_sampling(labels=next_op['cat_id'], logits=logits, counts=category_counts['cat_id'], ns_size=10)

optimizer = tf.train.AdamOptimizer()
optimizer_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_num in range(3):
        sess.run(dataset_iterator.initializer)
        try:
            for batch_num in itertools.count():
                l, _ = sess.run([loss, optimizer_step],
                                feed_dict={K.learning_phase(): 1})
                print(f'Epoch {epoch_num} batch {batch_num} loss {l}')
        except OutOfRangeError:
            pass

