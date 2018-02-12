from PIL import Image
import os
from tqdm import tqdm
import multiprocessing
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_HEIGHT = IMAGE_WIDTH = 299
THUMBNAIL_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
PROCESSED_PHOTOS_DIR = 'processed_photos'
PHOTOS_DIR = 'photos'
ROOT_PATH = '.'
PROCESSED_PHOTOS_PATH = os.path.join(ROOT_PATH, PROCESSED_PHOTOS_DIR)
PHOTOS_PATH = os.path.join(ROOT_PATH, PHOTOS_DIR)
NUM_PROCESSES = 100

def process_photo(filename_photo):
    background = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), "black")
    source_image = Image.open(os.path.join(PHOTOS_PATH, filename_photo)).convert("RGB")
    source_image.thumbnail(THUMBNAIL_SIZE)
    upperleft = (int((THUMBNAIL_SIZE[0] - source_image.size[0]) / 2),
                 int((THUMBNAIL_SIZE[1] - source_image.size[1]) / 2))
    background.paste(source_image, upperleft)

    background.save(os.path.join(PROCESSED_PHOTOS_PATH, filename_photo), 'JPEG')

def process_photos():
    if not os.path.exists(PROCESSED_PHOTOS_PATH):
        os.makedirs(PROCESSED_PHOTOS_PATH)
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    filename_photos_original = np.array(os.listdir(PHOTOS_PATH))
    filename_photos_processed = np.array(os.listdir(PROCESSED_PHOTOS_PATH))
    filename_photos_original = filename_photos_original[
        ~ np.isin(filename_photos_original, filename_photos_processed)]
    list(tqdm(pool.imap_unordered(process_photo, filename_photos_original), total=len(filename_photos_original)))

if __name__ == '__main__':
    process_photos()