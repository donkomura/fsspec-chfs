import numpy as np
from numpy.lib.format import write_array, read_array
import fsspec
import pytest
import os


@pytest.fixture(autouse=True)
def init():
    server = os.getenv("CHFS_SERVER")
    if not server:
        raise ValueError("CHFS server may not be up. Set CHFS_SERVER")
    yield

def test_file_rw():
    fs = fsspec.filesystem("chfs")
    with fs.open("zzz") as f:
        f.write(bytearray(b'zzzzzzzz'))
    del f
    with fs.open("zzz") as f:
        assert f.read() == b'zzzzzzzz'

def test_file_npy():
    fs = fsspec.filesystem("chfs")
    a = np.arange(3)
    with fs.open("pydata") as f:
        write_array(f, a)
    del f
    with fs.open("pydata") as f:
        b = read_array(f)
        assert np.array_equal(a, b)

def test_file_readinto():
    fs = fsspec.filesystem("chfs")
    with fs.open("zzz") as f:
        f.write(bytearray(b'zzzzzzzz'))
    del f
    with fs.open("zzz") as f:
        buf = bytearray(b'\x00') * len(b'zzzzzzzz')
        s = f.readinto(buf)
        assert s == len(b'zzzzzzzz')
        assert buf == b'zzzzzzzz'

def test_file_seek():
    fs = fsspec.filesystem("chfs")
    with fs.open("abc") as f:
        f.write(bytearray(b'abcdef'))
        f.seek(-3, os.SEEK_CUR)
        buf = f.read(3)
        assert buf == b'def'

def test_file_npload():
    fs = fsspec.filesystem("chfs")
    a = np.arange(100)
    with fs.open("pydata2") as f:
        np.save(f, a)
    del f
    with fs.open("pydata2") as f:
        b = np.load(f)
        print(b)
        assert np.array_equal(a, b)
