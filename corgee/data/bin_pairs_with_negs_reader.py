import copy
import logging

import numpy as np
import torch
from data.bin_pairs_reader import BinLPairsDataLoader, get_binlines_len
from main.main_utils import scatter_tensor_from_one
from numba import njit

logger = logging.getLogger(__name__)


@njit
def fill_dense_array_pairs_with_negs(data, densemat1, densemat2):
    offset, numrows, maxlen1 = 0, densemat1.shape[0], densemat1.shape[1]
    num_with_negatives, maxlen2 = densemat2.shape[1], densemat2.shape[2]
    for rownum in range(numrows):
        datal1 = data[offset]
        l1 = min(datal1, maxlen1)
        densemat1[rownum, :l1] = data[offset + 1 : offset + l1 + 1]
        offset += datal1 + 1

        for neg_idx in range(num_with_negatives):
            datal2 = data[offset]
            l2 = min(datal2, maxlen2)
            densemat2[rownum, neg_idx, :l2] = data[offset + 1 : offset + l2 + 1]
            offset += datal2 + 1


def read_pairs_with_negs(data, maxlen1, maxlen2, num_negatives):
    numrows = get_binlines_len(data) // (2 + num_negatives)
    densemat1 = np.zeros((numrows, maxlen1), dtype=np.uint16)
    densemat2 = np.zeros((numrows, 1 + num_negatives, maxlen2), dtype=np.uint16)
    fill_dense_array_pairs_with_negs(data, densemat1, densemat2)
    # another option is to store in int16 format and handle negative indices when vocab > 32k
    # but that would save 2x memory, but extra negative check in loading, risky
    return torch.from_numpy(densemat1.astype(np.int32)), torch.from_numpy(
        densemat2.astype(np.int32)
    )


class BinLPairsWithNegativesDataLoader(BinLPairsDataLoader):
    """
    Because of efficiency, loading will happen in only rank=0 worker, others will just collect data
    before starting
    """

    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.num_negatives = config["data"]["num_negatives"]
        super(BinLPairsWithNegativesDataLoader, self).__init__(
            config, rank, world_size, device
        )

    def _populate_file_contents_queue(self, file_path):
        # check whether pool has been stopped before starting file read
        if not self.stopped:
            feat1, feat2 = read_pairs_with_negs(
                self._read_file(file_path),
                self.maxlen1,
                self.maxlen2,
                self.num_negatives,
            )
        # check whether pool has been stopped before putting data into queue
        if not self.stopped:
            # put read data in queue, this waits when queue is full until someone reads from it
            self._file_contents_queue.put((file_path, feat1, feat2))

    def __iter__(self):
        while self.num_batches < self.num_steps:
            self.num_batches += 1
            feats1, feats2 = [], []
            if self.rank == 0:
                feats1, feats2 = self._train_batches_queue.get(timeout=600)
                self._train_batches_queue.task_done()
                if self.prefix1 is not None:
                    bsz1, numtok1 = feats1.shape
                    feats1 = torch.cat(
                        (self.prefix1[None, :].repeat(bsz1, 1), feats1), dim=1
                    )[:, :numtok1]
                if self.prefix2 is not None:
                    bsz2, num_ex, numtok2 = feats2.shape
                    feats2 = torch.cat(
                        (self.prefix2[None, None, :].repeat(bsz2, num_ex, 1), feats2),
                        dim=2,
                    )[:, :, :numtok2]
                if self.suffix1 is not None:
                    start_cols = torch.minimum(
                        (feats1 > 0).sum(1) + len(self.suffix1),
                        torch.tensor(self.maxlen1),
                    ) - len(self.suffix1)
                    start_rows = torch.arange(feats1.shape[0])
                    for _i in range(len(self.suffix1)):
                        feats1[start_rows, start_cols + _i] = self.suffix1[_i]
                if self.suffix2 is not None:
                    start_cols = torch.minimum(
                        (feats2 > 0).sum(2) + len(self.suffix2),
                        torch.tensor(self.maxlen2),
                    ) - len(self.suffix2)
                    batch_indices = (
                        torch.arange(feats2.size(0))
                        .unsqueeze(1)
                        .expand(-1, feats2.size(1))
                    )
                    dim2_indices = torch.arange(feats2.size(1)).expand(
                        feats2.size(0), -1
                    )
                    for _i in range(len(self.suffix2)):
                        feats2[
                            batch_indices, dim2_indices, start_cols - 1
                        ] = self.suffix2[_i]
                feats1 = list(feats1.to(device="cuda").tensor_split(self.world_size))
                feats2 = list(feats2.to(device="cuda").tensor_split(self.world_size))

            feats1_dest = torch.empty(
                (self.batch_size // self.world_size, self.maxlen1),
                device="cuda",
                dtype=torch.int32,
            )
            feats2_dest = torch.empty(
                (
                    self.batch_size // self.world_size,
                    self.num_negatives + 1,
                    self.maxlen2,
                ),
                device="cuda",
                dtype=torch.int32,
            )
            # feats1 and feats2 are now loaded and can be transferred to other ranks
            if self.world_size > 1:
                feats1, handler1 = scatter_tensor_from_one(feats1_dest, feats1)
                feats2, handler2 = scatter_tensor_from_one(feats2_dest, feats2)
                # to atleast overlap communication of both sides of features
                if handler1 is not None:
                    handler1.wait()
                if handler2 is not None:
                    handler2.wait()
            else:
                feats1 = feats1[0]
                feats2 = feats2[0]

            if self.cut_percentile is not None:

                def next_multiple(num, multiple):
                    num += multiple - 1
                    num -= num % multiple
                    return num

                bsz, num_ex, _ = feats2.shape
                percentile_pos1 = min(int(bsz * self.cut_percentile), bsz - 1)
                cut_feat1 = torch.sort((feats1 != self.padding_index).sum(1)).values[
                    percentile_pos1
                ]
                feats1 = feats1[
                    :, : next_multiple(cut_feat1, self.cut_percentile_multiple)
                ]
                percentile_pos2 = min(
                    int(bsz * num_ex * self.cut_percentile), bsz * num_ex - 1
                )
                cut_feat2 = torch.sort(
                    (feats2 != self.padding_index).sum(2).ravel()
                ).values[percentile_pos2]
                feats2 = feats2[
                    :, :, : next_multiple(cut_feat2, self.cut_percentile_multiple)
                ]
                if self.rank == 0:
                    logger.info(f"{feats1.shape[1]} {feats2.shape[2]}")

            feats1 = self.input_ids_to_dict_feat(feats1)
            feats2 = self.input_ids_to_dict_feat(feats2)
            batch_data = {"feats": {"text1": feats1, "text2": feats2}}
            batch_data["target"] = self.target
            batch_data["batch_size"] = self.batch_size // self.world_size
            yield batch_data


class BinLPairsWithNegativesDataLoaderMultitask:
    def __init__(self, config, rank, world_size, device, *args, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.num_steps, self.num_batches = int(config["num_steps"]), 0
        data_config = config["data"]
        self.datasets = list(data_config["files"].keys())
        self.sample_probs = np.array(
            [data_config["files"][d]["rate"] for d in self.datasets], dtype=np.float32
        )
        self.sample_probs /= self.sample_probs.sum()
        logger.info(f"datasets: {self.datasets}")
        logger.info(f"datasets: {self.sample_probs}")
        self.dls = {}
        for d in self.datasets:
            config_copy = copy.deepcopy(config)
            del config_copy["data"]["files"]
            config_copy["data"]["reader_queue_len"] = data_config["files"][d].get(
                "reader_queue_len", 2
            )
            config_copy["data"]["num_reader_threads"] = data_config["files"][d].get(
                "num_reader_threads", 1
            )
            config_copy["data"]["maxlen1"] = data_config["files"][d]["maxlen1"]
            config_copy["data"]["maxlen2"] = data_config["files"][d]["maxlen2"]
            config_copy["data"]["file_pattern"] = data_config["files"][d][
                "file_pattern"
            ]
            self.dls[d] = BinLPairsWithNegativesDataLoader(
                config_copy, rank, world_size, device, *args, **kwargs
            )

    def __iter__(self):
        self.dl_iters = [iter(self.dls[x]) for x in self.datasets]
        while self.num_batches < self.num_steps:
            self.num_batches += 1
            dataset_idx = np.random.choice(len(self.sample_probs), p=self.sample_probs)
            batch = next(self.dl_iters[dataset_idx])
            batch["dataset_idx"] = dataset_idx
            yield batch

    def join(self):
        for dl in self.dls.values():
            dl.join()

    def __len__(self):
        return self.num_steps

    def parse_steps(self, steps):
        assert isinstance(steps, int), "bin lines datareader needs integer steps"
        return steps

    def state_dict(self):
        return {
            "num_batches": self.num_batches,
            "dls": {k: v.state_dict() for k, v in self.dls.items()},
        }

    def load_state_dict(self, sd):
        self.num_batches = sd["num_batches"]
        for k, v in sd["dls"]:
            self.dls[k].load_state_dict(v)
