from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import os
import pickle
import sys


# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout


# #### Download scifact.zip dataset and unzip the dataset
# dataset = "scifact"
# model = "msmarco-distilbert-base-tas-b"
dataset = sys.argv[1]
model = sys.argv[2]
score_fn = sys.argv[3]  # dot or cos_sim
print(dataset, model)


url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = ''
data_path = util.download_and_unzip(url, out_dir)


out_path_pkl = os.path.join(out_dir, 'results', dataset, f'{model}_{score_fn}.pkl')
os.makedirs(os.path.dirname(out_path_pkl), exist_ok=True)


# Provide the data_path where scifact has been downloaded and unzipped
# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


# Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT(model), batch_size=64)
retriever = EvaluateRetrieval(model, score_function=score_fn)  # or "cos_sim" for cosine similarity


if os.path.exists(out_path_pkl):
    with open(out_path_pkl, 'rb') as f:
        results = pickle.load(f)
else:
    results = retriever.retrieve(corpus, queries)

    with open(out_path_pkl, 'wb') as f:
        pickle.dump(results, f)


# Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
