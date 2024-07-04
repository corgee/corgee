import yaml
import webdataset as wds
import tqdm.autonotebook as tqdm
import gzip
import json
import os
import re
import multiprocessing

conf = yaml.safe_load(open('config_data_pretrain.yaml').read())
input_dir = ''
output_dir = ''


def process_data(input):
    path_in, path_out, t1_col, t2_col = input
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
            fout.write(f"{t1}\t{t2}\n")
    return 1

for dataset in conf['datasets']:
    if dataset["name"] == "reddit_title_body":
        continue
    print(f'doing dataset {dataset["name"]}')
    urls = wds.shardlists.expand_urls(dataset['bucket'])
    paths_in = [url.replace('s3://', input_dir) for url in urls]
    paths_out = [url.replace('s3://', output_dir).replace('.gz', '') for url in urls]
    t1_col, t2_col = dataset['objective']['columns']
    
    # for path_in, path_out in tqdm.tqdm(zip(paths_in, paths_out)):
    #     process_data((path_in, path_out, t1_col, t2_col))

    inputs = [(pin, pout, t1_col, t2_col) for pin, pout in zip(paths_in, paths_out)]
    pool = multiprocessing.Pool(min(16, len(inputs)))
    res = pool.map(process_data, inputs)
    pool.close()
    pool.join()
