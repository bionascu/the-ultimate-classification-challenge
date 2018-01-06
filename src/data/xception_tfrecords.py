import io
import bson
import itertools
from os import path, environ

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
from sklearn.externals import joblib
from keras.applications.xception import Xception

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


def products_from_bson(filename):
    for product in bson.decode_file_iter(open(filename, 'rb')):
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
    features = xception.predict(images)
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


examples_per_tfrecord = 20 if environ.get('TEST_RUN') == 'true' else 20_000

products = products_from_bson(path.join(data_raw_dir, 'train_example.bson' if environ.get('TEST_RUN') == 'true' else 'train.bson'))
products_with_feats = itertools.chain.from_iterable(
    map(extract_features, batches_from(products, 64, allow_shorter=True)))
examples = map(make_example, products_with_feats)

total_count = 0
for batch_num, examples_batch in enumerate(batches_from(examples, examples_per_tfrecord, allow_shorter=True)):
    filename = path_expression_train.format(batch_num) \
        if batch_num % 10 == 0 \
        else path_expression_test.format(batch_num)
    with tf.python_io.TFRecordWriter(filename, options=options) as writer:
        count = 0
        for example in examples_batch:
            writer.write(example.SerializeToString())
            count += 1
    total_count += count
    print('Written {} examples into {} total examples {}'.format(count, path.basename(filename), total_count))
