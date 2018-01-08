import glob
import itertools
import datetime
import argparse
import pandas as pd
from os import path, getpid

from keras import backend as K
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tqdm import tqdm

from data.tfrecords_read import parse_example
from model import model
from model.loss_function import *
from utils import models_dir, data_processed_dir, logs_dir

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', dest='run_id', type=str,
                    default=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
parser.add_argument('--restore', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--train-batch-size', type=int, default=100)
parser.add_argument('--test-batch-size', type=int, default=100)

args = parser.parse_args()

checkpoint_dir = path.join(models_dir, args.run_id)
if args.restore and tf.train.latest_checkpoint(checkpoint_dir) is None:
    print(f'Could not restore {args.run_id}')
    exit(1)

print(f'Process pid: {getpid()}')

# Create train graph
with tf.name_scope('data_reader_train'):
    filenames = glob.glob(path.join(data_processed_dir, 'train_*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames, compression_type='ZLIB') \
        .shuffle(3 * args.train_batch_size) \
        .map(parse_example) \
        .batch(args.train_batch_size)
    train_dataset_iterator = dataset.make_initializable_iterator()
    train_next_batch = train_dataset_iterator.get_next()

logits = model(train_next_batch['features'])

correct_prediction = tf.equal(train_next_batch['cat_id'], tf.argmax(logits, axis=1))
train_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
counts = pd.read_pickle(path.join(data_processed_dir, 'cat_id_img_distrib.pickle'))
train_loss = negative_sampling(labels=train_next_batch['cat_id'], logits=logits,
                               counts=counts, ns_size=1000)

optimizer = tf.train.AdamOptimizer()
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer_step = optimizer.minimize(train_loss, global_step=global_step)

# Create test graph
with tf.name_scope('data_reader_test'):
    filenames = glob.glob(path.join(data_processed_dir, 'test_*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames, compression_type='ZLIB') \
        .map(parse_example) \
        .repeat() \
        .batch(args.test_batch_size)
    test_dataset_iterator = dataset.make_initializable_iterator()
    test_next_batch = test_dataset_iterator.get_next()

logits = model(test_next_batch['features'])

correct_prediction = tf.equal(test_next_batch['cat_id'], tf.argmax(logits, axis=1))
test_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=test_next_batch['cat_id'], logits=logits), name='loss')

with tf.name_scope('metrics'):
    train_acc_summary = tf.summary.scalar('train_acc', train_acc)
    test_acc_summary = tf.summary.scalar('test_acc', test_acc)
    train_loss_summary = tf.summary.scalar('train_loss', train_loss)
    test_loss_summary = tf.summary.scalar('test_loss', test_loss)

saver = tf.train.Saver(
    model.trainable_weights +
    [global_step] +
    [v for v in tf.global_variables() if 'Adam' in v.name],
    max_to_keep=2
)

with tf.Session() as sess:
    # Merge all the summaries and write them out to the logs folder
    train_merged = tf.summary.merge([train_acc_summary, train_loss_summary], name='train_summaries')
    test_merged = tf.summary.merge([test_acc_summary, test_loss_summary], name='test_summaries')

    summary_writer = tf.summary.FileWriter(path.join(logs_dir, args.run_id), sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(test_dataset_iterator.initializer)
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        gs = sess.run(global_step)
        print(f'Restored {args.run_id}, global step: {gs}')

    tqdm_progress = tqdm(unit=' examples')
    for epoch_num in range(args.epochs):
        sess.run(train_dataset_iterator.initializer)

        # Record summaries and train-set accuracy
        try:
            for batch_num in itertools.count(0):
                summary, a, l, _, gs = sess.run([train_merged, train_acc, train_loss, optimizer_step, global_step],
                                                feed_dict={K.learning_phase(): 1})
                summary_writer.add_summary(summary, gs)
                tqdm_progress.update(args.train_batch_size)
                tqdm_progress.set_postfix(epoch=f'{epoch_num}/{args.epochs}', batch=batch_num, accuracy=f'{a:.3%}', loss=f'{l:.6f}')

                # Record summaries and test-set accuracy
                if batch_num % 3 == 0:
                    summary, a, l, gs = sess.run([test_merged, test_acc, test_loss, global_step],
                                                 feed_dict={K.learning_phase(): 0})
                    summary_writer.add_summary(summary, gs)
                    tqdm_progress.write(f'Test epoch {epoch_num}\tgs {gs}:\tacc {a:.3%}\tloss {l:.6f}')

                if batch_num % 10:
                    save_path = saver.save(sess, path.join(checkpoint_dir, 'model'), global_step=global_step)
                    tqdm_progress.write(f'Model saved in {save_path}')
        except OutOfRangeError:
            pass

    summary, a, l, gs = sess.run([test_merged, test_acc, test_loss, global_step],
                                 feed_dict={K.learning_phase(): 0})
    summary_writer.add_summary(summary, gs)
    tqdm_progress.write(f'Test epoch {epoch_num}\tgs {gs}:\tacc {a:.3%}\tloss {l:.6f}')

    save_path = saver.save(sess, path.join(checkpoint_dir, 'model'), global_step=global_step)
    tqdm_progress.write(f'Model saved in {save_path}')

    tqdm_progress.close()
    summary_writer.close()
