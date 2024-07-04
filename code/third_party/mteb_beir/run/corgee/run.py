from mteb import MTEB
import yaml
import sys
import logging
from beir_utils import TOKENIZERS_MAP, BEIR_CACHE_DIR, MyModelAllGPU
from main.main_utils import read_config
from model.encoders import create_encoder


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


assert len(sys.argv) == 8, "incorrect number of args"
config_file = sys.argv[1]
model_file = sys.argv[2]
output_dir = sys.argv[3]
encoder_name = sys.argv[4]  # encoder name in config.yaml
tokenizer_name = sys.argv[5]  # defining some standard tokenizers in this file
max_per_gpu_bsz = int(sys.argv[6])  # 128
'''
for load balancing, tasks can be split into
    MSMARCO
    FEVER,ClimateFEVER
    ArguAna,CQADupstackAndroidRetrieval,CQADupstackEnglishRetrieval,CQADupstackGamingRetrieval,CQADupstackGisRetrieval,CQADupstackMathematicaRetrieval,CQADupstackPhysicsRetrieval,CQADupstackProgrammersRetrieval,CQADupstackStatsRetrieval,CQADupstackTexRetrieval,CQADupstackUnixRetrieval,CQADupstackWebmastersRetrieval,CQADupstackWordpressRetrieval,DBPedia,FiQA2018,HotpotQA,NFCorpus,NQ,QuoraRetrieval,SCIDOCS,SciFact,Touche2020,TRECCOVID
'''
tasks = sys.argv[7].split(',')
lang_list = ['en']

if encoder_name.startswith('compiled:'):
    suffix = "module._orig_mod."
    encoder_name = encoder_name[len('compiled:'):]
else:
    suffix = "module."

if encoder_name.startswith('notstrict:'):
    strict = False
    encoder_name = encoder_name[len('notstrict:'):]
else:
    strict = True

# read model config from experiment directory
config = read_config(config_file)
try:
    encoder_config = config['model']['encoders'][encoder_name]
except:
    raise f"encoder_name {encoder_name} should be in config"

# tokenizer should be in map
assert tokenizer_name in TOKENIZERS_MAP, f"tokenizer name {tokenizer_name} should be in templates"

# tweak encoder config, create encoder and start evaluation
encoder_config['load'] = yaml.safe_load(f'''path: {model_file}
strict: {strict}
transformations:
    - filter: "encoders.{encoder_name}.{suffix}"
    - replace_start: "encoders.{encoder_name}.{suffix}=>"''')
model = MyModelAllGPU(create_encoder(encoder_config, {}), max_per_gpu_bsz)

evaluation = MTEB(tasks=tasks, beir_cache_dir=BEIR_CACHE_DIR, task_langs=lang_list, beir_pretok=tokenizer_name)
results = evaluation.run(model, output_folder=output_dir)
