from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter


class DataLoaderWithPrefetch(DataLoader):
    def __init__(self, *args, prefetch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = prefetch_size if prefetch_size is not None else 2 * kwargs.get("num_workers", 0)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIterWithPrefetch(self)


class _MultiProcessingDataLoaderIterWithPrefetch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.prefetch_size = loader.prefetch_size

        super().__init__(loader)

        # Prefetch more items than the default 2 * self._num_workers
        assert self.prefetch_size >= 2 * self._num_workers
        for _ in range(loader.prefetch_size - 2 * self._num_workers):
            self._try_put_index()

    def _try_put_index(self):
        assert self._tasks_outstanding < self.prefetch_size
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
