import pandas as pd
import urllib.request
import os
from tqdm import tqdm

PHOTOS_DIR = 'photos'
ROOT_PATH = '.'

def download_data():
    photos_path = os.path.join(ROOT_PATH, PHOTOS_DIR)
    users_df = pd.read_csv('Users.csv', sep=';', dtype={'User_id': str})

    if not os.path.exists(photos_path):
        os.makedirs(photos_path)

    for (id, url) in tqdm(users_df[['User_id', 'Photo']][:100].values):
        urllib.request.urlretrieve(url, os.path.join(photos_path, id + '.jpg'))

if __name__ == '__main__':
    download_data()