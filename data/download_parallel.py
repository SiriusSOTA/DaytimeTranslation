import os
import json
import requests
from tqdm import tqdm
import multiprocessing


def func(inp):
    fname, url = inp
    if not url:
        print(fname + ' no url')
        return
    res = sorted([(url.lower().find('.jpg'), 'jpg'),
                 (url.lower().find('.jpeg'), 'jpg'),
                 (url.lower().find('.png'), 'png')])[-1][1]
    fname += res
    if os.path.isfile(fname):
        return #continue
    try:
        r = requests.get(url, allow_redirects=True, timeout=3)
        with open(fname, 'wb') as f:
            f.write(r.content)
    except:
        pass #print(url + ' failed')


pool = multiprocessing.Pool(8)

# request ids to be ignored
ban_list = [
    '051394c7-5884-4d38-a18e-9a822954052d',
    '7c58ac38-7d1c-4f19-b112-b5b43f50a6ef',
    '1a7a7474-9855-4677-834a-d7a167b206d1'
]

# path to save
dataset = '.'
print(dataset)

for fname in os.listdir(dataset):
    if 'serps' not in fname:
        continue
    with open(dataset + '/' + fname) as f:
        data = json.load(f)

    if '_' in fname:
        folder = dataset + '/' + fname.split('_')[-1] + '/'
    else:
        folder = dataset + '/images/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for d in data:
        if d['serpRequestExplained']['id'] in ban_list:
            continue
        subfolder = folder + d['serpRequestExplained']['id']
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder) 
        urls = d['serp-page']['parser-results']['components']
        urls = [('%s/%05d.' % (subfolder, c), i['image-url']) for c, i in enumerate(urls)]
        if not urls:
            continue
        print(urls[0])
        pool.map(func, urls)