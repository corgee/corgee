import os
import git
import requests
from mteb.abstasks import AbsTask, AbsTaskBitextMining, AbsTaskClassification, AbsTaskClustering,\
    AbsTaskPairClassification, AbsTaskReranking, AbsTaskRetrieval, AbsTaskSTS, AbsTaskSummarization


all_tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
all_tasks_cls = [
    cls()
    for cat_cls in all_tasks_categories_cls
    for cls in cat_cls.__subclasses__()
    if cat_cls.__name__.startswith("AbsTask")
]


def check_and_download_data(out_dir, dataset_name):
    dirpath = os.path.join(out_dir, dataset_name)
    if not os.path.isdir(dirpath):
        print(f"{dirpath} dir does not exist")
        zippath = os.path.join(out_dir, dataset_name+'.zip')
        if not os.path.exists(zippath):
            print(f"{zippath} file does not exist")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            if requests.get(url).status_code == 404:
                print(f'{dataset_name} not available on ukp website. skipping...')
                return
            os.system(f'wget -P {out_dir} {url}')
            # os.system(f'axel -avn 10 --output {out_dir} {url}')
        else:
            print(f"{zippath} file exists")
        print("unzipping")
        os.system(f'unzip -d "{out_dir}" "{zippath}"')
    else:
        print(f"{dirpath} dir exists")

# download beir datasets
OUT_DIR=""
os.makedirs(OUT_DIR, exist_ok=True)
ret_tasks = list(filter(lambda x: isinstance(x, AbsTaskRetrieval), all_tasks_cls))
for _task in ret_tasks:
    if 'beir_name' in _task.description:
        check_and_download_data(OUT_DIR, _task.description['beir_name'])
    else:
        print(f'skipping {_task.description["name"]}')


def check_and_clone_data(out_dir, dataset_name, revision):
    dirpath = os.path.join(out_dir, dataset_name)
    if not os.path.isdir(dirpath):
        split = dataset_name.split('/')
        if len(split) != 2:
            print(f'{dataset_name} not of the form user/repo')
            return
        user, repo = split
        if user != 'mteb':
            print(f'skipping non mteb user: {user}')
            return
        print(f"{dirpath} dir does not exist")
        repo_out_dir = os.path.join(out_dir, user)
        os.makedirs(repo_out_dir, exist_ok=True)
        git.Git(repo_out_dir).clone(f"https://huggingface.co/datasets/{user}/{repo}")
        if revision is not None:
            try:
                git.Repo(dirpath).git.checkout(revision)
            except:
                print('checkout failed')
                return
    else:
        print(f"{dirpath} dir exists")
        if revision is not None:
            if not git.Repo(dirpath).head.object.hexsha == revision:
                print('hash doesnt match with revision id')
                return

# clone all mteb dataset repos
OUT_DIR = ""
for clstask in [AbsTaskBitextMining, AbsTaskClassification, AbsTaskPairClassification, AbsTaskReranking,\
    AbsTaskSTS, AbsTaskSummarization, AbsTaskClustering, AbsTaskRetrieval]:
    tasks = list(filter(lambda x: isinstance(x, clstask), all_tasks_cls))
    for _task in tasks:
        if 'hf_hub_name' in _task.description:
            check_and_clone_data(OUT_DIR, _task.description['hf_hub_name'], _task.description.get('revision'))
        else:
            print(f'skipping {_task.description["name"]} no hf_hub_name')
