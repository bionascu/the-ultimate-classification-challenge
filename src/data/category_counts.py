import bson
from os import path

import pandas as pd
from sklearn.externals import joblib

from utils.paths import data_raw_dir, data_processed_dir


encoder_cat_1 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_1.pickle')))
encoder_cat_2 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_2.pickle')))
encoder_cat_3 = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_3.pickle')))
encoder_cat_id = joblib.load(path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))

categories = pd.read_csv(path.join(data_raw_dir, 'category_names.csv'))
categories['category_level1'] = encoder_cat_1.transform(categories.category_level1)
categories['category_level2'] = encoder_cat_2.transform(categories.category_level2)
categories['category_level3'] = encoder_cat_3.transform(categories.category_level3)
categories['category_id'] = encoder_cat_id.transform(categories.category_id)
categories.set_index('category_id', inplace=True)


def categories_from_bson(filename):
    for product in bson.decode_file_iter(open(filename, 'rb')):
        product_id = product['_id']
        category_id = encoder_cat_id.transform([int(product['category_id'])])[0]
        category_levels = categories.loc[category_id]
        yield {
            'prod_id': product_id,
            'cat_id': category_id,
            'cat_1': category_levels.category_level1,
            'cat_2': category_levels.category_level2,
            'cat_3': category_levels.category_level3,
        }


product_categories = pd.DataFrame(categories_from_bson(path.join(data_raw_dir, 'train_example.bson')),
                        columns=['prod_id', 'cat_id', 'cat_1', 'cat_2', 'cat_3'])

category_counts = {col_name: product_categories[col_name].value_counts() for col_name in product_categories.columns.values.tolist()[1:]}
