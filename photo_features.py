from keras.applications.inception_v3 import InceptionV3
from lib import *
import os
import pandas as pd
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tqdm import tqdm
from multiprocessing import Pool
import math
from keras.applications.inception_v3 import InceptionV3
num_batches = 32

if os.path.isfile('users_df'):
    users_df = pd.read_pickle('users_df')
else:
    users_df, _ = load_data()
    users_df['feature'] = None

users_df['feature'] = users_df['feature'].asobject
# users_df = pd.read_pickle('users_df')
model = VGGFace(include_top=False, input_shape=(299, 299, 3), pooling='avg')
# model = InceptionV3(include_top=False, input_shape=(299, 299, 3), pooling='avg', weights='imagenet')
# for layer in vgg_model.layers:
#     if hasattr(layer, 'trainable'):
#         layer.trainable = False

def extract_feature(user_ids):
    paths = map(lambda user_id: os.path.join('processed_photos', str(user_id) + '.jpg'), user_ids)
    imgs = [np.array(Image.open(path), dtype=np.float64) for path in paths]
    # img = np.expand_dims(img, axis=0)
    imgs = np.stack(imgs)
    imgs = utils.preprocess_input(imgs, version=1)  # or version=2
    features = model.predict(imgs)
    for i in range(len(user_ids)):
        users_df.set_value(user_ids[i], 'feature', features[i])

def main():
    # users_df, _ = load_data()
    # pool = Pool(2)
    # list(tqdm(pool.imap_unordered(extract_feature, np.array_split(users_df.index.values, num_batches)), total=len(users_df)//num_batches))
    # i = 0
    ids = users_df.index.values
    for split in tqdm(np.array_split(ids, math.ceil(len(ids)/num_batches))):
        extract_feature(split)
    # list(tqdm((extract_feature, np.array_split(users_df.index.values, num_batches)), total=len(users_df)//num_batches))

    # list(tqdm(map(extract_feature, users_df.index.values), total=len(users_df)))

    users_df.to_pickle('users_df')

    print('done')

if __name__ == '__main__':
    main()


# for i in range(len(users_df)):
#     img = Image.open(filepaths[i])
#     users_df.iloc[i, 'photo_feature'] = model.predict(np.array(img))
# print('done')