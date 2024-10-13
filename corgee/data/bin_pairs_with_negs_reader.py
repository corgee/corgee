import copy
import logging

import numpy as np
import torch
from data.bin_pairs_reader import BinLPairsDataLoader, BinLPairsDataLoaderMultitask, get_binlines_len
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
    densemat1 = np.zeros((numrows, maxlen1), dtype=data.dtype)
    densemat2 = np.zeros((numrows, 1 + num_negatives, maxlen2), dtype=data.dtype)
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


class BinLPairsWithNegativesDataLoaderMultitask(BinLPairsDataLoaderMultitask):
    SingleDatasetClass = BinLPairsWithNegativesDataLoader
