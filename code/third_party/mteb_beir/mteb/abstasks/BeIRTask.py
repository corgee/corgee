import os

import datasets
import pickle
import numpy as np
from numba import njit

from .AbsTask import AbsTask
from beir_utils import load_tokenized, prepare_feat_beir, get_stats
import logging

logger = logging.getLogger(__name__)

class BeIRTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        default_cache_dir = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
        self.beir_cache_dir = kwargs.get('beir_cache_dir', default_cache_dir)
        self.beir_pretok_query = kwargs.get('beir_pretok_query', kwargs.get('beir_pretok', None))
        self.beir_pretok_doc = kwargs.get('beir_pretok_doc', kwargs.get('beir_pretok', None))
        logger.info(f'Using cache dir: {self.beir_cache_dir}')

    def unload_data(self, eval_splits=None, **kwargs):
        for split in eval_splits:
            del self.corpus[split], self.queries[split], self.relevant_docs[split]
        self.data_loaded = False

    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        
        try:
            from beir import util
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        USE_HF_DATASETS = False

        from beir.datasets.data_loader import GenericDataLoader 
        logger.info("Using GenericDataLoader for BeIR")

        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        dataset = self.description["beir_name"]
        dataset, sub_dataset = dataset.split("/") if "cqadupstack" in dataset else (dataset, None)

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            if not os.path.isdir(os.path.join(self.beir_cache_dir, dataset)):
                util.download_and_unzip(url, self.beir_cache_dir)
            data_path = os.path.join(self.beir_cache_dir, dataset)
            data_path = os.path.join(data_path, sub_dataset) if sub_dataset else data_path
            if self.beir_pretok_query is not None and self.beir_pretok_doc is not None:
                prepare_feat_beir(data_path, split, self.beir_pretok_query, self.beir_pretok_doc)
                tok_dir_query = os.path.join(data_path, self.beir_pretok_query)
                tok_dir_doc = os.path.join(data_path, self.beir_pretok_doc)
                self.queries[split] = load_tokenized(os.path.join(tok_dir_query, f'queries.{split}.pretok'), self.beir_pretok_query)
                self.corpus[split] = load_tokenized(os.path.join(tok_dir_doc, 'corpus.pretok'), self.beir_pretok_doc)
                with open(os.path.join(data_path, f'qrels.{split}.pkl'), 'rb') as fin:
                    self.relevant_docs[split] = pickle.load(fin)
            else:
                self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                    data_folder=data_path
                ).load(split=split)
        # get_stats(self.corpus, self.queries, self.relevant_docs)
        self.data_loaded = True
