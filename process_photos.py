from PIL import Image
import os
from tqdm import tqdm
import multiprocessing
import numpy as np
from PIL import ImageFile, ImageOps
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_HEIGHT = IMAGE_WIDTH = 224
THUMBNAIL_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
PROCESSED_PHOTOS_DIR = 'processed_photos'
PHOTOS_DIR = 'photos'
ROOT_PATH = '.'
PROCESSED_PHOTOS_PATH = os.path.join(ROOT_PATH, PROCESSED_PHOTOS_DIR)
PHOTOS_PATH = os.path.join(ROOT_PATH, PHOTOS_DIR)
NUM_PROCESSES = 100
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

def process_photo(filename_photo):
    img = cv2.imread(os.path.join(PHOTOS_PATH, filename_photo))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=6, minSize=(200, 200));

    if (len(faces) == 0):
        source_image = Image.open(os.path.join(PHOTOS_PATH, filename_photo)).convert("RGB")
        resize_and_save(filename_photo, source_image)
    else:
        if len(faces) > 1:
            wh = [f[2] + f[3] for f in faces]
            faces = [faces[np.argmax(wh)]]
        (x, y, w, h) = faces[0]
        img = img[y:y + w, x:x + h]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_and_save(filename_photo, Image.fromarray(img))

def resize_and_save(filename_photo, source_image):
    img = ImageOps.fit(source_image, THUMBNAIL_SIZE)

    img.save(os.path.join(PROCESSED_PHOTOS_PATH, filename_photo), 'JPEG')

def process_photos():
    if not os.path.exists(PROCESSED_PHOTOS_PATH):
        os.makedirs(PROCESSED_PHOTOS_PATH)
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    filename_photos_original = np.array(os.listdir(PHOTOS_PATH))
    filename_photos_processed = np.array(os.listdir(PROCESSED_PHOTOS_PATH))
    filename_photos_original = filename_photos_original[
        ~ np.isin(filename_photos_original, filename_photos_processed)][:5000]
    list(tqdm(pool.imap_unordered(process_photo, filename_photos_original), total=len(filename_photos_original)))

if __name__ == '__main__':
    process_photos()