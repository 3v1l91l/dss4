from PIL import Image
import os
from tqdm import tqdm
import multiprocessing
import numpy as np
from PIL import ImageFile
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
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=7, minSize=(200, 200));

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
    background = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), "black")
    source_image.thumbnail(THUMBNAIL_SIZE)
    if (~(np.array(source_image.size) == THUMBNAIL_SIZE)).any():
        max_im_size = max(source_image.size)
        background2 = Image.new('RGB', (max_im_size, max_im_size), "black")
        upperleft = (int((max_im_size - source_image.size[0]) / 2),
                     int((max_im_size - source_image.size[1]) / 2))
        background2.paste(source_image, upperleft)
        source_image = background2.resize(THUMBNAIL_SIZE)
    upperleft = (int((THUMBNAIL_SIZE[0] - source_image.size[0]) / 2),
                 int((THUMBNAIL_SIZE[1] - source_image.size[1]) / 2))
    background.paste(source_image, upperleft)

    background.save(os.path.join(PROCESSED_PHOTOS_PATH, filename_photo), 'JPEG')

def process_photos():
    if not os.path.exists(PROCESSED_PHOTOS_PATH):
        os.makedirs(PROCESSED_PHOTOS_PATH)
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    filename_photos_original = np.array(os.listdir(PHOTOS_PATH))[:3000]
    # filename_photos_original =  np.array(['3002361535.jpg'])
    filename_photos_processed = np.array(os.listdir(PROCESSED_PHOTOS_PATH))
    filename_photos_original = filename_photos_original[
        ~ np.isin(filename_photos_original, filename_photos_processed)]
    list(tqdm(pool.imap_unordered(process_photo, filename_photos_original), total=len(filename_photos_original)))

if __name__ == '__main__':
    process_photos()