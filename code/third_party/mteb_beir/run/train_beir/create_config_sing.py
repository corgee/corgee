import sys
import os
import yaml

out_file = sys.argv[1]
config_paths = sys.argv[2].split()
target = sys.argv[3]
workspace = sys.argv[4]
sku = sys.argv[5]
preemptible = sys.argv[6]

BEIR_ENCODER_NAME = "text1"
BEIR_TOKENIZER_NAME = "bert_base_uncased_cls101"
BEIR_BATCH_SIZE = 64

image = "acpt-rocm5.4.2_ubuntu20.04_py3.8_pytorch_2.0.0" if "MI200" in sku else ("acpt-2.1.0-cuda12.1" if "A100" in sku else "ptca-1.13.1-cuda11.7")
# image = "acpt-rocm5.4.2_ubuntu20.04_py3.8_pytorch_2.0.0" if "MI200" in sku else "acpt-2.1.0-cuda11.8"

header = f"""target:
  service: sing
  name: {target}
  workspace_name: {workspace}
environment:
  image: amlt-sing/{image}
  image_setup:
    - sudo apt-get update -y
    - sudo apt-get install zlib1g-dev -y
code:
  local_dir: $CONFIG_DIR/..
jobs:
"""

job_template = f"""- name: {{config_path}}:+allbeir
  priority: Medium
  sku: {sku}
  command:
    - cd remote/blobfuse2; source setup.sh; cd ../..
    - source remote/setup_sing.sh
    - export NCCL_IB_DISABLE=1
    - nvidia-smi topo -m
    - source run.sh {{config_path}}
{{model_eval_line}}
  submit_args:
    max_run_duration_seconds: 432000

"""

model_eval_line_template = f"    - source code/third_party/mteb_beir/run/corgee/run.sh {{beir_config_path}} {{beir_model_path}} {{beir_output_path}} {BEIR_ENCODER_NAME} {BEIR_TOKENIZER_NAME} {BEIR_BATCH_SIZE} MSMARCO,FEVER,ClimateFEVER,ArguAna,CQADupstackAndroidRetrieval,CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,DBPedia,FiQA2018,HotpotQA,NFCorpus,NQ,QuoraRetrieval,SCIDOCS,SciFact,Touche2020,TRECCOVID"
def process_model_path(model_path, prefix='step'):
    base_name = os.path.basename(model_path)
    assert base_name.startswith(prefix) and base_name.endswith('.pt')
    assert base_name[len(prefix):-len('.pt')].isdigit()
    models_dirname = os.path.dirname(model_path)
    assert os.path.basename(models_dirname) == 'models'
    models_configpath = os.path.join(os.path.dirname(models_dirname), 'config.yaml')
    output_path = os.path.join(models_dirname, f'beir_{base_name[:-len(".pt")]}')
    return models_configpath, output_path

def get_model_eval_line(config_path):
    config = yaml.safe_load(open(config_path).read())
    num_steps = config['exec']['train']['data']['num_steps'] - 1
    assert config_path.startswith('configs/') and config_path.endswith('.yaml')
    assert "${EXP_NAME}" in config['exec']['output_dir']
    out_dir = config['exec']['output_dir'].replace("${EXP_NAME}", config_path[len('configs/'):-len('.yaml')])
    beir_model_path = os.path.join(out_dir, 'models', f'step{num_steps}.pt')
    beir_config_path, beir_output_path = process_model_path(beir_model_path)
    model_eval_line = model_eval_line_template.format(beir_config_path=beir_config_path, beir_model_path=beir_model_path, 
    beir_output_path=beir_output_path)
    
    if 'ema' in config['exec']['train']['forward_backward']:
        beir_model_path = os.path.join(out_dir, 'models', f'ema{num_steps}.pt')
        beir_config_path, beir_output_path = process_model_path(beir_model_path, prefix='ema')
        # first evaluate ema then vanilla model
        model_eval_line = model_eval_line_template.format(beir_config_path=beir_config_path, beir_model_path=beir_model_path, 
    beir_output_path=beir_output_path)+'\n'+model_eval_line

    return model_eval_line

with open(out_file, 'w') as fout:
    fout.write(header)
    for config_path in config_paths:
        model_eval_line = get_model_eval_line(config_path)
        fout.write(job_template.format(config_path=config_path, model_eval_line=model_eval_line))
