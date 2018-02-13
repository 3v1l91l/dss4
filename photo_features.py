from keras.applications.inception_v3 import InceptionV3
from lib import *
import os
import pandas as pd
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tqdm import tqdm

users_df, _ = load_data()
filepaths = list(map(lambda x: os.path.join('processed_photos', str(x) + '.jpg'), users_df.index.values))

vgg_model = VGGFace(include_top=False, input_shape=(299, 299, 3), pooling='avg') # pooling: None, avg or max


users_df['feature'] = 0

users_df['feature'] = users_df['feature'].astype(object)

for i in tqdm(range(len(filepaths))):
    img = np.array(Image.open(filepaths[i]), dtype=np.float64)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=1)  # or version=2
    features = vgg_model.predict(img)
    users_df.iat[i, users_df.columns.get_loc('feature')] = features
users_df.to_pickle('users_df')

print('done')
#
# for i in range(len(users_df)):
#     img = Image.open(filepaths[i])
#     users_df.iloc[i, 'photo_feature'] = model.predict(np.array(img))
# print('done')