import sys
import os
import yaml


out_file = sys.argv[1]
config_paths = sys.argv[2].split()
target = sys.argv[3]
vc = sys.argv[4]
sku = sys.argv[5]
preemptible = sys.argv[6]

BEIR_ENCODER_NAME = "text1"
BEIR_TOKENIZER_NAME = "bert_base_uncased_cls101"
BEIR_BATCH_SIZE = 64

header = f"""target:
  service: amlk8s
  name: {target}
  vc: {vc}
environment:
  image: python:3.8
  registry: docker.io
code:
  local_dir: $CONFIG_DIR/..
jobs:
"""

job_template = f"""- name: {{config_path}}:+allbeir
  priority: Medium
  sku: {sku}
  command:
    - cd remote/blobfuse2; source setup.sh; cd ../..
    - source remote/setup_itp.sh
    - source run.sh {{config_path}}
    - source code/third_party/mteb_beir/run/corgee/run.sh {{beir_config_path}} {{beir_model_path}} {{beir_output_path}} {BEIR_ENCODER_NAME} {BEIR_TOKENIZER_NAME} {BEIR_BATCH_SIZE} MSMARCO,FEVER,ClimateFEVER,ArguAna,CQADupstackAndroidRetrieval,CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,DBPedia,FiQA2018,HotpotQA,NFCorpus,NQ,QuoraRetrieval,SCIDOCS,SciFact,Touche2020,TRECCOVID
"""


def process_model_path(model_path):
    base_name = os.path.basename(model_path)
    assert base_name.startswith('step') and base_name.endswith('.pt')
    assert base_name[len('step'):-len('.pt')].isdigit()
    models_dirname = os.path.dirname(model_path)
    assert os.path.basename(models_dirname) == 'models'
    models_configpath = os.path.join(os.path.dirname(models_dirname), 'config.yaml')
    output_path = os.path.join(models_dirname, f'beir_{base_name[:-len(".pt")]}')
    return f"beir_eval_{model_path.replace('/', '_')}", models_configpath, output_path


def get_model_path(config_path):
    config = yaml.safe_load(open(config_path).read())
    num_steps = config['exec']['train']['data']['num_steps'] - 1
    assert config_path.startswith('configs/') and config_path.endswith('.yaml')
    assert "${EXP_NAME}" in config['exec']['output_dir']
    out_dir = config['exec']['output_dir'].replace("${EXP_NAME}", config_path[len('configs/'):-len('.yaml')])
    return os.path.join(out_dir, 'models', f'step{num_steps}.pt')


with open(out_file, 'w') as fout:
    fout.write(header)
    for config_path in config_paths:
        beir_model_path = get_model_path(config_path)
        _, beir_config_path, beir_output_path = process_model_path(beir_model_path)
        fout.write(job_template.format(config_path=config_path, beir_config_path=beir_config_path, beir_model_path=beir_model_path, beir_output_path=beir_output_path))
