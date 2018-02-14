from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

# model = VGGFace(include_top=False, input_shape=(299, 299, 3), pooling='avg')
model = InceptionV3(include_top=False, input_shape=(299, 299, 3), pooling='avg', weights='imagenet')
img1 = Image.open(os.path.join('processed_photos', '2995240981.jpg'))
img1 = np.array(img1)[np.newaxis,:]
img1 = img1.astype(np.float64)
img2 = Image.open(os.path.join('processed_photos', '2994529591.jpg'))
img2 = np.array(img2)[np.newaxis,:]
img2 = Image.open(os.path.join('processed_photos', '2993855363.jpg'))
img2 = np.array(img2)[np.newaxis,:]
img2 = img2.astype(np.float64)
#
# img1 = utils.preprocess_input(img1, version=1)  # or version=2
# img2 = utils.preprocess_input(img1, version=1)  # or version=2

features1 = model.predict(img1)
features2 = model.predict(img2)
print(cosine_similarity(features1, features2))
# model.summary()
