import bson
from os import path, environ

import pandas as pd
from sklearn.externals import joblib

from utils.paths import data_raw_dir, data_processed_dir


encoder_cat_id = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))
categories = pd.read_pickle(path.join(data_processed_dir, 'categories.pickle'))


def product_distrib_from_bson(filename):
    for product in bson.decode_file_iter(open(filename, 'rb')):
        product_id = product['_id']
        category_id = encoder_cat_id.transform([int(product['category_id'])])[0]
        category_levels = categories.loc[category_id]
        yield {
                'prod_id': product_id,
                'cat_id': category_id,
                'cat_1': category_levels.cat_1,
                'cat_2': category_levels.cat_2,
                'cat_3': category_levels.cat_3
            }


def image_distrib_from_bson(filename):
    img_id = 0
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
                'img_id': img_id
            }
            img_id += 1


# Create product distribution (by category)
product_distrib = pd.DataFrame.from_records(
    product_distrib_from_bson(path.join(data_raw_dir, 'train_example.bson' if
    environ.get('TEST_RUN') == 'true' else 'train.bson')))

for col_name in product_distrib.columns.drop('prod_id'):
    category_counts = product_distrib[col_name].value_counts()
    category_counts.to_pickle(path.join(data_processed_dir,
                                        f'{col_name}_prod_distrib.pickle'))


# Create image distribution (by category)
image_distrib = pd.DataFrame.from_records(
    image_distrib_from_bson(path.join(data_raw_dir, 'train_example.bson' if
    environ.get('TEST_RUN') == 'true' else 'train.bson')))

for col_name in image_distrib.columns.drop('prod_id', 'img_id'):
    category_counts = image_distrib[col_name].value_counts()
    category_counts.to_pickle(path.join(data_processed_dir,
                                        f'{col_name}_img_distrib.pickle'))
