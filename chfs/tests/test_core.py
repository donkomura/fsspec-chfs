import fsspec
import pytest
import os
import numpy as np
import time

@pytest.fixture(autouse=True)
def init():
    server = os.getenv("CHFS_SERVER")
    if not server:
        raise ValueError("CHFS server may not be up. Set CHFS_SERVER")
    yield

def test_fsspec_local():
    fsspec.filesystem("file")

def test_fsspec_invalid():
    with pytest.raises(ValueError):
        fsspec.filesystem("foo")

def test_fsspec_chfs():
    fsspec.filesystem("chfs")

def join(dir, p):
    return os.path.join(dir, p)

def test_local_mkdir(tmpdir):
    f = fsspec.filesystem("file")
    foo = join(tmpdir, 'foo')
    f.mkdir(foo)
    assert f.exists(foo)

def test_chfs_mkdir(tmpdir):
    f = fsspec.filesystem("chfs")
    foo = join(tmpdir, 'foo')
    f.mkdir(foo)
    assert f.exists(foo)

def test_local_mkdir_rec(tmpdir):
    f = fsspec.filesystem("file")
    ab = join(tmpdir, 'foo/a/b')
    f.mkdir(ab, create_parents=True)
    with pytest.raises(FileNotFoundError):
        abcde = join(tmpdir, 'foo/a/b/c/d/e')
        f.mkdir(abcde, create_parents=False)

def test_chfs_mkdir_rec(tmpdir):
    f = fsspec.filesystem("chfs")
    ab = join(tmpdir, 'foo/a/b')
    f.mkdir(ab, create_parents=True)
    with pytest.raises(NotImplementedError):
        abcde = join(tmpdir, 'foo/a/b/c/d/e')
        f.mkdir(abcde, create_parents=False)

def test_local_touch(tmpdir):
    f = fsspec.filesystem("file")
    bar = join(tmpdir, 'bar')
    f.touch(bar)
    assert f.exists(bar)

def test_chfs_touch(tmpdir):
    f = fsspec.filesystem("chfs")
    bar = join(tmpdir, 'bar')
    f.touch(bar)
    assert f.exists(bar)


def test_local_rm(tmpdir):
    f = fsspec.filesystem("file")
    baz = join(tmpdir, 'baz')
    f.touch(baz)
    assert f.exists(baz)
    f.rm(baz)
    assert not f.exists(baz)

def test_chfs_rm(tmpdir):
    f = fsspec.filesystem("chfs")
    baz = join(tmpdir, 'baz')
    f.touch(baz)
    assert f.exists(baz)
    f.rm(baz)
    assert not f.exists(baz)

def test_local_rm_rec(tmpdir):
    f = fsspec.filesystem("file")
    baz = join(tmpdir, 'baz')
    abcd = join(tmpdir, 'baz/a/b/c/d')
    f.mkdirs(abcd)
    f.rm(baz, recursive=True)
    assert not f.exists(baz)

def test_local_rm_list(tmpdir):
    f = fsspec.filesystem("file")
    ps = [join(tmpdir, 'a'), join(tmpdir, 'b')]
    for p in ps:
        f.touch(p)
    f.rm(ps)
    for p in ps:
        assert not f.exists(p)

def test_local_cat(tmpdir):
    f = fsspec.filesystem("file")
    dog = join(tmpdir, 'dog')
    f.touch(dog)
    assert f.cat(dog) == b''

def test_chfs_cat(tmpdir):
    f = fsspec.filesystem("chfs")
    dog = join(tmpdir, 'dog')
    f.touch(dog)
    assert f.cat(dog) == b''

def test_local_cat_list(tmpdir):
    f = fsspec.filesystem("file")
    dog1, dog2 = join(tmpdir, 'dog1'), join(tmpdir, 'dog2')
    f.touch(dog1)
    f.touch(dog2)
    assert f.cat([dog1, dog2]) == {dog1: b'', dog2: b''}

def test_chfs_cat_list(tmpdir):
    f = fsspec.filesystem("chfs")
    dog1, dog2 = join(tmpdir, 'dog1'), join(tmpdir, 'dog2')
    f.touch(dog1)
    f.touch(dog2)
    assert f.cat([dog1, dog2]) == {dog1: b'', dog2: b''}


def test_local_pipe(tmpdir):
    f = fsspec.filesystem("file")
    crow = join(tmpdir, 'crow')
    f.pipe(crow, b'caw caw')
    assert f.cat(crow) == b'caw caw'

def test_chfs_pipe(tmpdir):
    f = fsspec.filesystem("chfs")
    crow = join(tmpdir, 'crow')
    data = bytearray(b'caw caw')
    f.pipe(crow, data)
    assert f.cat(crow) == b'caw caw'

def test_chfs_pipe_read_only(tmpdir):
    f = fsspec.filesystem("chfs")
    crow = join(tmpdir, 'crow')
    f.pipe(crow, bytearray(b'caw caw'))
    assert f.cat(crow) == b'caw caw'

def test_chfs_pipe_bytes(tmpdir):
    f = fsspec.filesystem("chfs")
    crow = join(tmpdir, 'crow')
    f.pipe(crow, bytes(b'caw caw'))
    assert f.cat(crow) == b'caw caw'

def test_chfs_pipe_ndarray_uint8(tmpdir):
    f = fsspec.filesystem("chfs")
    p = join(tmpdir, 'npydata')
    data = np.arange(3, dtype=np.uint8)
    f.pipe(p, data)
    assert np.array_equal(f.cat(p), data)

def test_chfs_pipe_ndarray_int8(tmpdir):
    f = fsspec.filesystem("chfs")
    p = join(tmpdir, 'npydata')
    data = np.arange(3, dtype=np.int8)
    with pytest.raises(ValueError):
        f.pipe(p, data)

def test_chfs_pipe_ndarray_uint32(tmpdir):
    f = fsspec.filesystem("chfs")
    p = join(tmpdir, 'npydata')
    data = np.arange(3, dtype=np.uint32)
    with pytest.raises(ValueError):
        f.pipe(p, data)

def test_local_pipe_dict(tmpdir):
    f = fsspec.filesystem("file")
    d = {
        join(tmpdir, 'p1'): b'p1',
        join(tmpdir, 'p2'): b'p2',
        join(tmpdir, 'p3'): b'p3'
    }
    f.pipe(d)
    for key in d:
        assert f.cat(key) == d[key]

def test_chfs_pipe_dict(tmpdir):
    f = fsspec.filesystem("chfs")
    d = dict()
    d[join(tmpdir, 'p1')] = bytearray(b'p1')
    d[join(tmpdir, 'p2')] = bytearray(b'p2')
    d[join(tmpdir, 'p3')] = bytearray(b'p3')
    f.pipe(d)
    for key in d:
        assert f.cat(key) == d[key]

def test_chfs_pipe_dict_read_only(tmpdir):
    f = fsspec.filesystem("chfs")
    d = {
        join(tmpdir, 'p1'): b'p1',
        join(tmpdir, 'p2'): b'p2',
        join(tmpdir, 'p3'): b'p3'
    }
    f.pipe(d)
    for key in d:
        assert f.cat(key) == d[key]

def test_local_mkdirs(tmpdir):
    f = fsspec.filesystem("file")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc)
    assert f.exists(cdc)
    f.mkdirs(cdc, exist_ok=True)
    assert f.exists(cdc)

def test_chfs_mkdirs(tmpdir):
    f = fsspec.filesystem("chfs")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc)
    assert f.exists(cdc)
    f.mkdirs(cdc, exist_ok=True)
    assert f.exists(cdc)

def test_local_pipe_file(tmpdir):
    f = fsspec.filesystem("file")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc, exist_ok=True)
    cdcv = join(tmpdir, 'cat/dog/crow/voice')
    f.pipe_file(cdcv, b'mew woof caw')
    assert f.cat(cdcv) == b'mew woof caw'

def test_chfs_pipe_file(tmpdir):
    f = fsspec.filesystem("chfs")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc, exist_ok=True)
    cdcv = join(tmpdir, 'cat/dog/crow/voice')
    data = bytearray(b'mew woof caw')
    f.pipe_file(cdcv, data)
    assert f.cat(cdcv) == data

def test_chfs_pipe_file_read_only(tmpdir):
    f = fsspec.filesystem("chfs")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc, exist_ok=True)
    cdcv = join(tmpdir, 'cat/dog/crow/voice')
    f.pipe_file(cdcv, bytearray(b'mew woof caw'))
    assert f.cat(cdcv) == bytearray(b'mew woof caw')

def test_chfs_pipe_file_bytes(tmpdir):
    f = fsspec.filesystem("chfs")
    cdc = join(tmpdir, 'cat/dog/crow')
    f.mkdirs(cdc, exist_ok=True)
    cdcv = join(tmpdir, 'cat/dog/crow/voice')
    f.pipe_file(cdcv, bytes(b'mew woof caw'))
    assert f.cat(cdcv) == bytes(b'mew woof caw')

def test_local_find(tmpdir):
    f = fsspec.filesystem("file")
    a, b = join(tmpdir, 'a'), join(tmpdir, 'b')
    f.touch(a)
    f.touch(b)
    assert f.find(tmpdir) == [a, b]

def test_local_isfile(tmpdir):
    f = fsspec.filesystem("file")
    a = join(tmpdir, 'a')
    f.touch(a)
    assert not f.isfile(tmpdir)
    assert f.isfile(a)

def test_chfs_isfile(tmpdir):
    f = fsspec.filesystem("chfs")
    a = join(tmpdir, 'a')
    f.touch(a)
    assert not f.isfile(tmpdir)
    assert f.isfile(a)

def test_local_info(tmpdir):
    f = fsspec.filesystem("file")
    foo = join(tmpdir, 'foo')
    f.mkdir(foo)
    info = f.info(foo)
    assert info['type'] == 'directory'
    assert not info['islink']

def test_chfs_info(tmpdir):
    f = fsspec.filesystem("chfs")
    foo = join(tmpdir, 'foo')
    f.mkdir(foo)
    info = f.info(foo)
    assert info['type'] == 'directory'
    assert not info['islink']

def test_chfs_time():
    s = time.time()
    TRIAL = 1000000
    for trial in range(TRIAL):
        f = fsspec.filesystem("chfs")
        del f

    print('init&term time=', (time.time() - s) / TRIAL)
