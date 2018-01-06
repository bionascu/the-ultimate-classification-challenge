import io
import bson
import itertools
import sys
from os import path, environ

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
from sklearn.externals import joblib
from keras.applications.xception import Xception
from tqdm import tqdm

from utils import data_raw_dir, data_processed_dir, batches_from

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
path_expression_train = path.join(data_processed_dir, 'train_{}.tfrecord')
path_expression_test = path.join(data_processed_dir, 'test_{}.tfrecord')

categories = pd.read_pickle(path.join(data_processed_dir, 'categories.pickle'))
encoder_cat_1 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_1.pickle')))
encoder_cat_2 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_2.pickle')))
encoder_cat_3 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_3.pickle')))
encoder_cat_id = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))

xception = Xception(include_top=False, weights='imagenet')
xception_batch_size = 512 if environ.get('TEST_RUN') == 'true' else 2048

examples_per_tfrecord = 20 if environ.get('TEST_RUN') == 'true' else 20000
skip_first_records = int(sys.argv[1])
n_records = 20


def products_from_bson(filename):
    for product in itertools.islice(bson.decode_file_iter(open(filename, 'rb')),
                                    start=skip_first_records * examples_per_tfrecord,
                                    stop=skip_first_records * examples_per_tfrecord + n_records * examples_per_tfrecord):
        product_id = product['_id']
        category_id = encoder_cat_id.transform([int(product['category_id'])])[0]
        category_levels = categories.loc[category_id]
        for product_picture in product['imgs']:
            yield {
                'prod_id': product_id,
                'cat_id': category_id,
                'cat_1': category_levels.cat_1,
                'cat_2': category_levels.cat_2,
                'cat_3': category_levels.cat_3,
                'img': imread(io.BytesIO(product_picture['picture']))
            }


def extract_features(products):
    images = np.stack((p['img'] for p in products), axis=0)
    features = xception.predict(images, batch_size=xception_batch_size)
    return [{**p, 'features': feats} for p, feats in zip(products, features)]


def make_example(product) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(feature={
        'prod_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[product['prod_id']])),
        'cat_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[product['cat_id']])),
        'cat_1': tf.train.Feature(int64_list=tf.train.Int64List(value=[product['cat_1']])),
        'cat_2': tf.train.Feature(int64_list=tf.train.Int64List(value=[product['cat_2']])),
        'cat_3': tf.train.Feature(int64_list=tf.train.Int64List(value=[product['cat_3']])),
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=product['features'].flatten())),
    }))


bson_file = 'train_example.bson' if environ.get('TEST_RUN') == 'true' else 'train.bson'
print(f'Processing {bson_file}\n'
      f'Skipping {skip_first_records * examples_per_tfrecord}\n'
      f'Xception will process {xception_batch_size} images at the time\n'
      f'Creating records of {examples_per_tfrecord} examples each\n')

products = products_from_bson(path.join(data_raw_dir, bson_file))
products_with_feats = itertools.chain.from_iterable(
    map(extract_features, batches_from(products, xception_batch_size, allow_shorter=True)))
examples = map(make_example, products_with_feats)

count = 0
filename = None
writer = None

tqdm_examples = tqdm(examples, unit=' examples')
for example_num, example in enumerate(tqdm_examples):
    if example_num % examples_per_tfrecord == 0:
        if writer:
            writer.close()
        batch_num = example_num // examples_per_tfrecord
        filename = path_expression_test.format(batch_num) if batch_num % 10 == 0 \
            else path_expression_train.format(batch_num)
        tqdm_examples.write('Writing to {}'.format(path.basename(filename)))
        writer = tf.python_io.TFRecordWriter(filename, options=options)
    writer.write(example.SerializeToString())
