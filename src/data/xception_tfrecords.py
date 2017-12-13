import io
import bson
import itertools
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from keras.applications.xception import Xception

from utils import data_raw_dir, data_processed_dir, batches_from

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
path_expression = path.join(data_processed_dir, 'train{}.tfrecord')

categories = pd.read_csv(path.join(data_raw_dir, 'category_names.csv'))

encoder_cat_1 = LabelEncoder()
encoder_cat_2 = LabelEncoder()
encoder_cat_3 = LabelEncoder()
encoder_cat_id = LabelEncoder()

categories['category_level1'] = encoder_cat_1.fit_transform(categories.category_level1)
categories['category_level2'] = encoder_cat_2.fit_transform(categories.category_level2)
categories['category_level3'] = encoder_cat_3.fit_transform(categories.category_level3)
categories['category_id'] = encoder_cat_id.fit_transform(categories.category_id)

joblib.dump(encoder_cat_1, path.join(path.join(data_processed_dir, 'encoder_cat_1.pickle')))
joblib.dump(encoder_cat_2, path.join(path.join(data_processed_dir, 'encoder_cat_2.pickle')))
joblib.dump(encoder_cat_3, path.join(path.join(data_processed_dir, 'encoder_cat_3.pickle')))
joblib.dump(encoder_cat_id, path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))

categories.set_index('category_id', inplace=True)

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
                'cat_1': category_levels.category_level1,
                'cat_2': category_levels.category_level2,
                'cat_3': category_levels.category_level3,
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


examples_per_tfrecord = 20

products = products_from_bson(path.join(data_raw_dir, 'train_example.bson'))
products_with_feats = itertools.chain.from_iterable(
    map(extract_features, batches_from(products, 64, allow_shorter=True)))
examples = map(make_example, products_with_feats)

total_count = 0
for batch_num, examples_batch in enumerate(batches_from(examples, examples_per_tfrecord, allow_shorter=True)):
    filename = path_expression.format(batch_num)
    with tf.python_io.TFRecordWriter(filename, options=options) as writer:
        count = 0
        for example in examples_batch:
            writer.write(example.SerializeToString())
            count += 1
    total_count += count
    print(f'Written {count} examples into {path.basename(filename)}, total examples {total_count}')
