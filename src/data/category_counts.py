import bson
from os import path, environ

import pandas as pd
from sklearn.externals import joblib

from utils.paths import data_raw_dir, data_processed_dir

encoder_cat_1 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_1.pickle')))
encoder_cat_2 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_2.pickle')))
encoder_cat_3 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_3.pickle')))
encoder_cat_id = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))

categories = pd.read_pickle(path.join(data_processed_dir, 'categories.pickle'))

category_columns = ['prod_id', 'cat_id', 'cat_1', 'cat_2', 'cat_3', ]

def categories_from_bson(filename):
    for product in bson.decode_file_iter(open(filename, 'rb')):
        product_id = product['_id']
        category_id = encoder_cat_id.transform([int(product['category_id'])])[0]
        category_levels = categories.loc[category_id]
        yield [
            product_id,
            category_id,
            category_levels.category_level1,
            category_levels.category_level2,
            category_levels.category_level3
        ]


product_categories = pd.DataFrame.from_records(
    categories_from_bson(path.join(data_raw_dir, 'train_example.bson' if environ.get('DRY_RUN') == 'true' else 'train.bson')),
    columns=category_columns)

for col_name in product_categories.columns.drop('prod_id'):
    category_counts = product_categories[col_name].value_counts()
    category_counts.to_pickle(path.join(data_processed_dir, f'{col_name}_counts.pickle'))
