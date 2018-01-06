from os import path

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from utils.paths import data_raw_dir, data_processed_dir

categories = pd.read_csv(path.join(data_raw_dir, 'category_names.csv'))

print(categories.nunique().to_frame('Category counts'))

encoder_cat_id = LabelEncoder()
encoder_cat_1 = LabelEncoder()
encoder_cat_2 = LabelEncoder()
encoder_cat_3 = LabelEncoder()

categories['cat_id'] = encoder_cat_id.fit_transform(categories.category_id)
categories['cat_1'] = encoder_cat_1.fit_transform(categories.category_level1)
categories['cat_2'] = encoder_cat_2.fit_transform(categories.category_level2)
categories['cat_3'] = encoder_cat_3.fit_transform(categories.category_level3)
categories.set_index('cat_id', inplace=True)
categories.to_pickle(path.join(path.join(data_processed_dir, 'categories.pickle')))

joblib.dump(encoder_cat_1, path.join(path.join(data_processed_dir, 'encoder_cat_1.pickle')))
joblib.dump(encoder_cat_2, path.join(path.join(data_processed_dir, 'encoder_cat_2.pickle')))
joblib.dump(encoder_cat_3, path.join(path.join(data_processed_dir, 'encoder_cat_3.pickle')))
joblib.dump(encoder_cat_id, path.join(path.join(data_processed_dir, 'encoder_cat_id.pickle')))

print('Done')
