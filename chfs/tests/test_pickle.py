import os
import cloudpickle
import fsspec
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def init():
    server = os.getenv("CHFS_SERVER")
    if not server:
        raise ValueError("CHFS server may not be up. Set CHFS_SERVER")
    yield

def test_pickle_chfs():
    fs = fsspec.filesystem("chfs", foo="bar")
    def save_fs():
        with fs.open('tmp/foo5') as f:
            f.write(bytearray(b'zzzzzzzz'))

    pickled_func = cloudpickle.dumps(save_fs)

    del save_fs
    print(pickled_func)
    func = cloudpickle.loads(pickled_func)
    func()

