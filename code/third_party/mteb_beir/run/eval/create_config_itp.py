import sys
import os


out_file = sys.argv[1]
model_paths = sys.argv[2:]
ENCODER_NAME = "text1"
TOKENIZER_NAME = "bert_base_uncased_cls101"
BATCH_SIZE = 64

header = """target:
  service: amlk8s
  name: ${TARGET_NAME}
  vc: ${VC}
environment:
  image: python:3.8
  registry: docker.io
code:
  local_dir: $CONFIG_DIR/..
jobs:
"""

job_template = """- name: {0}:all
  priority: Medium
  sku: ${{SKU}}
  command:
    - cd remote/blobfuse2; source setup.sh; cd ../..
    - source remote/setup_itp.sh
    - source code/third_party/mteb_beir/run/corgee/run.sh {1} {2} {3} {4} {5} {6} MSMARCO,FEVER,ClimateFEVER,ArguAna,CQADupstackAndroidRetrieval,CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,DBPedia,FiQA2018,HotpotQA,NFCorpus,NQ,QuoraRetrieval,SCIDOCS,SciFact,Touche2020,TRECCOVID
"""


def process_model_path(model_path):
    base_name = os.path.basename(model_path)
    assert base_name.startswith('step') and base_name.endswith('.pt')
    assert base_name[len('step'):-len('.pt')].isdigit()
    models_dirname = os.path.dirname(model_path)
    assert os.path.basename(models_dirname) == 'models'
    models_configpath = os.path.join(os.path.dirname(models_dirname), 'config.yaml')
    assert os.path.exists(models_configpath)
    output_path = os.path.join(models_dirname, f'beir_{base_name[:-len(".pt")]}')
    return f"beir_eval_{model_path.replace('/', '_')}", models_configpath, output_path


with open(out_file, 'w') as fout:
    fout.write(header)
    for model_path in model_paths:
        exp_name, config_path, output_path = process_model_path(model_path)
        fout.write(job_template.format(exp_name, config_path, model_path, output_path, ENCODER_NAME, TOKENIZER_NAME, BATCH_SIZE))
