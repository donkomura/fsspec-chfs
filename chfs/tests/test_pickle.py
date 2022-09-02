import cloudpickle
import fsspec
import numpy as np


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

