import glob
import itertools
import datetime
from os import path, environ

from keras import backend as K
from tensorflow.python.framework.errors_impl import OutOfRangeError

from data.tfrecords_read import parse_example
from model import model
from model.loss_function import *
from utils import models_dir, data_processed_dir, logs_dir

EPOCHS = 4 if environ.get('TEST_RUN') == 'true' else 500
TRAIN_BATCH_SIZE = 10 if environ.get('TEST_RUN') == 'true' else 2_000
TEST_BATCH_SIZE = 20 if environ.get('TEST_RUN') == 'true' else 100_000


def create_computational_graph(data_path, batch_size):
    with tf.name_scope('data_reader'):
        filenames = glob.glob(data_path)
        dataset = tf.data.TFRecordDataset(filenames, compression_type='ZLIB') \
            .shuffle(3 * batch_size) \
            .map(parse_example) \
            .batch(batch_size)
        dataset_iterator = dataset.make_initializable_iterator()
        next_batch = dataset_iterator.get_next()

    logits = model(next_batch['features'])

    correct_prediction = tf.equal(next_batch['cat_id'], tf.argmax(logits, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=next_batch['cat_id'], logits=logits), name='loss')

    # negative loss - need to load categories
    # loss = negative_sampling(labels=next_batch['cat_id'], logits=logits,
    #                          counts=category_counts['cat_id'], ns_size=10)

    return dataset_iterator, next_batch, acc, loss


run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# Create train graph
train_dataset_iterator, train_next_batch, train_acc, train_loss = \
    create_computational_graph(path.join(data_processed_dir, 'train_*.tfrecord'), batch_size=TRAIN_BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer_step = optimizer.minimize(train_loss, global_step=global_step)

# Create test graph
test_dataset_iterator, test_next_batch, test_acc, test_loss = \
    create_computational_graph(path.join(data_processed_dir, 'test_*.tfrecord'), batch_size=TEST_BATCH_SIZE)

with tf.name_scope('metrics'):
    train_acc_summary = tf.summary.scalar('train_acc', train_acc)
    test_acc_summary = tf.summary.scalar('test_acc', test_acc)
    train_loss_summary = tf.summary.scalar('train_loss', train_loss)
    test_loss_summary = tf.summary.scalar('test_loss', test_loss)

saver = tf.train.Saver(model.trainable_weights)

with tf.Session() as sess:
    # Merge all the summaries and write them out to the logs folder
    train_merged = tf.summary.merge([train_acc_summary, train_loss_summary], name='train_summaries')
    test_merged = tf.summary.merge([test_acc_summary, test_loss_summary], name='test_summaries')

    summary_writer = tf.summary.FileWriter(path.join(logs_dir, run_id), sess.graph)

    # train_writer = tf.summary.FileWriter(path.join('../logs', 'train'), sess.graph)
    # test_writer = tf.summary.FileWriter(path.join('../logs', 'test'))

    sess.run(tf.global_variables_initializer())

    for epoch_num in range(EPOCHS):
        sess.run(train_dataset_iterator.initializer)
        sess.run(test_dataset_iterator.initializer)

        # Record summaries and train-set accuracy
        try:
            for batch_num in itertools.count():
                summary, a, l, _, gs = sess.run([train_merged, train_acc, train_loss, optimizer_step, global_step],
                                                feed_dict={K.learning_phase(): 1})
                summary_writer.add_summary(summary, gs)
                print(f'Train {epoch_num} ({batch_num}):\tacc {a:.3%}\tloss {l:.6f}')
        except OutOfRangeError:
            pass

        # Record summaries and test-set accuracy
        try:
            for batch_num in itertools.count():
                summary, a, l, gs = sess.run([test_merged, test_acc, test_loss, global_step],
                                             feed_dict={K.learning_phase(): 0})
                summary_writer.add_summary(summary, gs)
                print(f'Test {epoch_num} ({batch_num}):\tacc {a:.3%}\tloss {l:.6f}')
        except OutOfRangeError:
            pass

        save_path = saver.save(sess, path.join(models_dir, run_id, 'model'), global_step=global_step)
        print(f'Model saved in {save_path}')

    summary_writer.close()
