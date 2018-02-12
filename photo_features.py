from keras.applications.inception_v3 import InceptionV3
from lib import *
import os
import pandas as pd
from PIL import Image

model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling='avg')
users_df, _ = load_data()

filepaths = list(map(lambda x: os.path.join('processed_photos', str(x) + '.jpg'), users_df.index.values))
imgs = list(map(Image.open, filepaths))
for i in range(len(users_df)):
    users_df.iloc[i, 'photo_feature'] = model.predict(imgs[i])
print('done')