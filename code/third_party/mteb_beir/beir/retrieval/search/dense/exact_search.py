import os
import numpy as np
import logging
import torch
import pickle
from typing import Dict, Optional
from beir_utils import TokenizedFeat
from .. import BaseSearch
from .util import cos_sim, dot_score
from main.main_utils import exact_nns, pq_exact_nns

logger = logging.getLogger(__name__)


from sklearn.preprocessing import normalize
import faiss
import scipy.sparse as sp


# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearchNew(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               save_path: Optional[str] = None,
               return_results: bool = True,
               **kwargs) -> Dict[str, Dict[str, float]]:
        if score_function != "cos_sim":
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        logger.info("Sorting Corpus by document length (Longest first)...")
        if isinstance(next(iter(corpus.values())), TokenizedFeat):
            corpus_ids = sorted(corpus, key=lambda k: corpus[k].attention_mask.sum(), reverse=True)
        else:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)


        cache_dir = kwargs.get('cache_dir')
        print(cache_dir)
        corpus_embeddings, query_embeddings = None, None

        if cache_dir is not None:
            if os.path.exists(os.path.join(cache_dir, 'corpus.npy')):
                corpus_embeddings = np.load(os.path.join(cache_dir, 'corpus.npy'))
            if os.path.exists(os.path.join(cache_dir, 'query.npy')):
                query_embeddings = np.load(os.path.join(cache_dir, 'query.npy'))
        
        if query_embeddings is None:
            query_embeddings = self.model.encode_queries(
                queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, **kwargs)
        if corpus_embeddings is None:
            corpus = [corpus[cid] for cid in corpus_ids]
            corpus_embeddings = self.model.encode_corpus(
                corpus, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, **kwargs)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            if not os.path.exists(os.path.join(cache_dir, 'corpus.npy')):
                np.save(os.path.join(cache_dir, 'corpus.npy'), corpus_embeddings)
            if not os.path.exists(os.path.join(cache_dir, 'query.npy')):
                np.save(os.path.join(cache_dir, 'query.npy'), query_embeddings)

        if kwargs.get('pq_config', None) is not None:
            res_smat = pq_exact_nns(query_embeddings, corpus_embeddings, kwargs['pq_config'], K=1005)
        else:
            res_smat = exact_nns(query_embeddings, corpus_embeddings, {'space': 'cosine'}, K=1005)
        for qnum, qid in enumerate(query_ids):
            q_results = {}
            indices = res_smat[qnum].indices
            data = res_smat[qnum].data
            order = data.argsort()[::-1]
            indices, data = indices[order], data[order]
            for cidx, c_distx in zip(indices, data):
                cid = corpus_ids[cidx]
                if cid == qid:
                    continue
                if len(q_results) == 1000:
                    break
                q_results[cid] = c_distx.item()
            self.results[qid] = q_results

        if save_path is not None:
            raise NotImplementedError
        if not return_results:
            return

        return self.results


# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearchOld(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               save_path: Optional[str] = None,
               return_results: bool = True,
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, **kwargs)

        logger.info("Sorting Corpus by document length (Longest first)...")

        if isinstance(next(iter(corpus.values())), TokenizedFeat):
            corpus_ids = sorted(corpus, key=lambda k: corpus[k].attention_mask.sum(), reverse=True)
        else:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        # because we remove qid == cid cases, could store one extra to accomodate
        # but alternatively, we cleanup any such matches before storing in buffer
        top_k_store = top_k
        BUFFER_SIZE = 10000 if len(query_ids) > 100000 else 50000
        query_topk_buffer_idx = torch.zeros(len(query_ids), top_k_store+BUFFER_SIZE, dtype=torch.int32)
        query_topk_buffer_values = torch.zeros(len(query_ids), top_k_store+BUFFER_SIZE, dtype=torch.float32)-torch.inf

        itr = range(0, len(corpus), self.corpus_chunk_size)
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor,
                **kwargs)

            query_ids_map = {x: i for i, x in enumerate(query_ids)}

            for subcorpus_split_start in range(0, len(sub_corpus_embeddings), BUFFER_SIZE):
                # Compute similarites using either cosine-similarity or dot product
                subcorpus_split = sub_corpus_embeddings[subcorpus_split_start: subcorpus_split_start+BUFFER_SIZE]
                cos_scores = self.score_functions[score_function](query_embeddings, subcorpus_split)
                cos_scores[torch.isnan(cos_scores)] = -1

                subcorpus_split_ids = corpus_ids[corpus_start_idx+subcorpus_split_start:corpus_start_idx+subcorpus_split_start+len(subcorpus_split)]
                matched_locs = [(query_ids_map[x], i) for i, x in enumerate(subcorpus_split_ids) if x in query_ids_map]
                if len(matched_locs) > 0:
                    cos_scores[[x[0] for x in matched_locs], [x[1] for x in matched_locs]] = -1

                query_topk_buffer_values[:, top_k_store:top_k_store+len(subcorpus_split)] = cos_scores
                subcorpus_split_ids = torch.arange(corpus_start_idx+subcorpus_split_start, corpus_start_idx+subcorpus_split_start+len(subcorpus_split))
                query_topk_buffer_idx[:, top_k_store:top_k_store+len(subcorpus_split)] = subcorpus_split_ids

                # Get top-k values
                query_topk_buffer_values[:, :top_k_store], top_indices = torch.topk(query_topk_buffer_values, top_k_store, dim=1, largest=True, sorted=return_sorted)
                query_topk_buffer_idx[:, :top_k_store] = query_topk_buffer_idx.gather(1, top_indices)
                query_topk_buffer_idx[:, top_k_store:], query_topk_buffer_values[:, top_k_store:] = 0, -torch.inf

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            results = {
                'query_topk_buffer_idx': query_topk_buffer_idx[:, :top_k_store],
                'query_topk_buffer_values': query_topk_buffer_values[:, :top_k_store]
            }
            torch.save(results, os.path.join(save_path, 'res.pt'))
            with open(os.path.join(save_path, 'query_ids.pkl'), "wb") as fp:
                pickle.dump(query_ids, fp)
            with open(os.path.join(save_path, 'corpus_ids.pkl'), "wb") as fp:
                pickle.dump(corpus_ids, fp)

        if not return_results:
            return

        for i, qid in enumerate(query_ids):
            for j in range(top_k):
                self.results[qid][corpus_ids[query_topk_buffer_idx[i, j]]] = query_topk_buffer_values[i, j].item()

        return self.results 


DenseRetrievalExactSearch = DenseRetrievalExactSearchNew