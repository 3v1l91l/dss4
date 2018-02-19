import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd

tf.__version__

PATH = os.getcwd()

LOG_DIR = PATH + '/embedding-logs'
# metadata = os.path.join(LOG_DIR, 'metadata2.tsv')

# %%
users_df = pd.read_pickle('users_df')
users_df = users_df[:1500]
# users_df = users_df[~users_df['feature'].isnull()].iloc[:400]
data_path = PATH + '/processed_photos'
img_data = []
for img in users_df.index.values:
    input_img = cv2.imread(data_path + '/' + str(img)+'.jpg')
    input_img_resize = cv2.resize(input_img, (100, 100))
    img_data.append(input_img_resize)

print('loaded photos')
img_data = np.array(img_data)
# users_df = users_df[np.isin(users_df.index.values, ids)]
# %%
# feature_vectors =
# feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
# print("feature_vectors_shape:", feature_vectors.shape)
# print("num of images:", feature_vectors.shape[0])
# print("size of individual feature vector:", feature_vectors.shape[1])
feature_vectors = np.stack(users_df['vgg_face'])
features = tf.Variable(feature_vectors, name='features')

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    # data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


# %%
sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)
# scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

# %%
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_4_classes.tsv')
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)