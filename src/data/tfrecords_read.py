import glob
import itertools
from os import path

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from utils.paths import data_processed_dir


def parse_example(example_proto):
    features = {
        'prod_id': tf.FixedLenFeature([], tf.int64),
        'cat_id': tf.FixedLenFeature([], tf.int64),
        'cat_1': tf.FixedLenFeature([], tf.int64),
        'cat_2': tf.FixedLenFeature([], tf.int64),
        'cat_3': tf.FixedLenFeature([], tf.int64),
        'features': tf.FixedLenFeature([6, 6, 2048], tf.float32),
    }
    example = tf.parse_single_example(example_proto, features)
    return example


if __name__ == '__main__':
    filenames = glob.glob(path.join(data_processed_dir, '*.tfrecord'))
    dataset = (
        tf.data.TFRecordDataset(filenames, compression_type='ZLIB')
            .map(parse_example)
            .batch(20)
    )
    dataset_iterator = dataset.make_initializable_iterator()
    next_op = dataset_iterator.get_next()

    with tf.Session() as sess:
        sess.run(dataset_iterator.initializer)

        try:
            for batch_num in itertools.count():
                next_batch = sess.run(next_op)
                print(f'Batch number {batch_num}')
                for k, v in next_batch.items():
                    print(f'  {k}: {v.shape}')
        except OutOfRangeError:
            pass
