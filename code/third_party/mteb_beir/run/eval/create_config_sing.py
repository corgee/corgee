import sys
import os


out_file = sys.argv[1]
model_paths = sys.argv[2].split()
target = sys.argv[3]
workspace = sys.argv[4]
sku = sys.argv[5]
preemptible = sys.argv[6]

ENCODER_NAME = "text1"
TOKENIZER_NAME = "bert_base_uncased_cls101"
BATCH_SIZE = 64

image = "acpt-rocm5.4.2_ubuntu20.04_py3.8_pytorch_2.0.0" if "MI200" in sku else "acpt-pytorch-1.12.1-cuda11.6"

header = f"""target:
  service: sing
  name: {target}
  workspace_name: {workspace}
environment:
  image: amlt-sing/{image}
  image_setup:
    - sudo apt-get update
    - sudo apt-get install zlib1g-dev
code:
  local_dir: $CONFIG_DIR/..
jobs:
"""

job_template = f"""- name: {{0}}:all
  priority: Medium
  sku: {sku}
  command:
    - cd remote/blobfuse2; source setup.sh; cd ../..
    - source remote/setup_sing.sh
    - source code/third_party/mteb_beir/run/corgee/run.sh {{1}} {{2}} {{3}} {ENCODER_NAME} {TOKENIZER_NAME} {BATCH_SIZE} MSMARCO,FEVER,ClimateFEVER,ArguAna,CQADupstackAndroidRetrieval,CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,DBPedia,FiQA2018,HotpotQA,NFCorpus,NQ,QuoraRetrieval,SCIDOCS,SciFact,Touche2020,TRECCOVID
  submit_args:
    max_run_duration_seconds: 432000

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
        fout.write(job_template.format(exp_name, config_path, model_path, output_path))
