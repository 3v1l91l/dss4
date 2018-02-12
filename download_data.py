import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
from urllib.request import urlretrieve
from urllib.error import URLError
from lib import load_data

PHOTOS_DIR = 'photos'
ROOT_PATH = '.'
PHOTOS_PATH = os.path.join(ROOT_PATH, PHOTOS_DIR)
NUM_PROCESSES = 100

def process_url(id_url):
    try:
        save_photo_path = os.path.join(PHOTOS_PATH, str(id_url[0]) + '.jpg')
        urlretrieve(id_url[1], save_photo_path)
    except URLError:
        print('URLError for id: {}, url: '.format(id_url[0], id_url[1]))

def download_data():
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    users_df, _ = load_data()

    if not os.path.exists(PHOTOS_PATH):
        os.makedirs(PHOTOS_PATH)
    existing_photos = [x.split('.')[0] for x in os.listdir(PHOTOS_PATH) if not x.startswith('.')]

    users_df.drop(existing_photos, inplace=True)
    urls = users_df['Photo'].values
    users_ids = users_df.index.values
    list(tqdm(pool.imap_unordered(process_url, zip(users_ids, urls)), total=len(users_ids)))

if __name__ == '__main__':
    download_data()