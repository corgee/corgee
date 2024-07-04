import yaml
import webdataset as wds
import tqdm.autonotebook as tqdm
import gzip
import json
import os
import re
import multiprocessing
import random


conf = yaml.safe_load(open('config_data_ft.yaml').read())
num_negatives = 7
input_dir = ''
output_dir = ''


def process_data(input):
    path_in, path_out, t1_col, t2_col, neg_col = input
    if os.path.exists(path_out):
        print(f'{path_out} exists continuing...')
        return
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with gzip.open(path_in,'rt') as fin, open(path_out, 'w') as fout:
        lines = fin.readlines()
        for line in lines:
            j = json.loads(line)
            t1 = re.sub(r'\s+', ' ', j[t1_col])
            t2 = re.sub(r'\s+', ' ', j[t2_col])
            feats = [t1, t2]
            feats.extend(list(map(lambda x: re.sub(r'\s+', ' ', x), random.sample(j[neg_col], num_negatives))))
            feats = '\t'.join(feats)
            fout.write(f"{feats}\n")


for dataset in conf['datasets']:
    print(f'doing dataset {dataset["name"]}')
    urls = wds.shardlists.expand_urls(dataset['bucket'])
    paths_in = [url.replace('s3://', input_dir) for url in urls]
    paths_out = [url.replace('s3://', output_dir).replace('.gz', '') for url in urls]
    t1_col, t2_col, neg_col = dataset['objective']['columns']

    inputs = [(pin, pout, t1_col, t2_col, neg_col) for pin, pout in zip(paths_in, paths_out)]
    # process_data(inputs[0])
    # break
    pool = multiprocessing.Pool(min(8, len(inputs)))
    res = pool.map(process_data, inputs)
    pool.close()
    pool.join()
