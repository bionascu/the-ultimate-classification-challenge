import itertools


def batches_from(iterable, batch_size, allow_shorter=True) -> list:
    next_batch = list(itertools.islice(iterable, batch_size))
    while next_batch and (allow_shorter or len(next_batch) == batch_size):
        yield next_batch
        next_batch = list(itertools.islice(iterable, batch_size))
