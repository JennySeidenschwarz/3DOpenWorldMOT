import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist
import os
from collections import defaultdict

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)



class DistributedSeqSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self.num_log_ids = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]

        # get per replica paths and their length
        self.per_replica_list = list()
        log_ids = list(range(len(self.dataset.data))) #list(self.dataset.data)
        max_len = list()
        for i in range(self.num_replicas):
            replica_list = log_ids[i*self.num_log_ids:(i+1)*self.num_log_ids]
            self.per_replica_list.append(replica_list)
            max_len.append(len(replica_list))
        
        # get maximum length replica paths and expected total size of datset with padding
        self.num_samples = max(max_len)
        self.total_size = self.num_replicas * self.num_samples

    def __iter__(self) -> Iterator[T_co]:
        # add extra samples to make it evenly divisible --> padding
        for i in range(self.num_replicas):
            if len(self.per_replica_list[i]) < self.num_samples:
                padding_size = self.num_samples - len(self.per_replica_list[i])
                self.per_replica_list[i] += self.per_replica_list[i][-padding_size:]
            assert len(self.per_replica_list[i]) == self.num_samples
        
        # flatten list
        indices = [p for replica_list in self.per_replica_list for p in replica_list]
        
        # double check of len of indices matches total size
        assert len(indices) == self.total_size

        # subsample indices per replica
        indices = indices[self.rank*int(self.total_size/self.num_replicas):(
            self.rank+1)*int(self.total_size/self.num_replicas)]
        
        # double check that len indices matches num samples per replica
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
