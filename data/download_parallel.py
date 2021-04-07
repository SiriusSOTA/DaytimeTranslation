import os
from pathlib import Path
import json
import multiprocessing

import cv2
import pandas as pd
import requests
from tqdm.auto import tqdm


def func(inp):
    fname, url = inp
    if not url:
        print(fname + ' no url')
        return
    res = sorted([(url.lower().find('.jpg'), 'jpg'),
                 (url.lower().find('.jpeg'), 'jpg'),
                 (url.lower().find('.png'), 'png')])[-1][1]
    fname += res
#     print(fname)
    if os.path.isfile(fname):
        return #continue
    try:
        r = requests.get(url, allow_redirects=True, timeout=5)
        if len(r.content) < 1024:
            return
    except:
        print(url + ' failed')
        return
    with open(fname, 'wb') as f:
        f.write(r.content)


def download_photos(fname, folder, ban_list):
    pool = multiprocessing.Pool(8)

    data = pd.read_csv(fname)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
    urls = []
    for d in tqdm(data.values):
        subfolder, fname = d[1].split('/')
        if subfolder in ban_list:
            continue
        subfolder = folder + subfolder
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder) 
        
        urls.append((folder + d[1], d[2]))

    pool.map(func, urls)
    
    
# request ids to be ignored
ban_list = [
        '051394c7-5884-4d38-a18e-9a822954052d',
        '7c58ac38-7d1c-4f19-b112-b5b43f50a6ef',
        '1a7a7474-9855-4677-834a-d7a167b206d1']
folder = 'images/'
fname = 'links.csv'

download_photos(fname, folder, ban_list)

# Clean useless files
data_path = Path('images/')
filenames = list(str(p) for p in data_path.glob('**/*.jpg')) + list(str(p) for p in data_path.glob('**/*.png'))

ban_lst = []
for fname in tqdm(filenames):
    try:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        ban_lst.append(fname)
        
print(len(ban_lst))
        
for fname in tqdm(ban_lst):
    os.remove(fname)
