import yaml
import webdataset as wds
import multiprocessing
from azure.datalake.store import core, lib
from cryptography.fernet import Fernet


# conf = yaml.safe_load(open('config_data_pretrain.yaml').read())
conf = yaml.safe_load(open('config_data_ft.yaml').read())
output_dir = ''
output_dir_cosmos = ''


def process_data(input):
    path_in, path_out, adl = input
    adl.put(path_in, path_out)
    print(f'{path_out} done')
    return


def decrypt(passkey, encrypted):
    fernet = Fernet(passkey)
    return fernet.decrypt(encrypted).decode()


def connect_adl():
    # can expose these parameters as parameters, but should be fixed for considerable future
    raise NotImplementedError
    passkey = ""
    encrypted = ""
    principal_token = lib.auth(
        tenant_id="",
        client_secret=decrypt(passkey, encrypted),
        client_id="",
    )
    return core.AzureDLFileSystem(token=principal_token, store_name="")


for dataset in conf['datasets']:
    print(f'doing dataset {dataset["name"]}')
    urls = wds.shardlists.expand_urls(dataset['bucket'])
    paths_out = [url.replace('s3://', output_dir).replace('.gz', '') for url in urls]
    paths_upload = [url.replace('s3://', output_dir_cosmos).replace('.gz', '') for url in urls]

    adl = connect_adl()

    inputs = [(pin, pout, adl) for pin, pout in zip(paths_out, paths_upload)]
    pool = multiprocessing.Pool(min(8, len(inputs)))
    res = pool.map(process_data, inputs)
    pool.close()
    pool.join()
