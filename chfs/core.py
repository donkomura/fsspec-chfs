from fsspec.spec import AbstractFileSystem
import io
import os
import stat
import logging
import pychfs

logger = logging.getLogger("fsspec-chfs")

class CHFSStubClient(AbstractFileSystem):
    """FSSpec-compatible wrapper of CHFS.
    """
    protocol = "chfs_stub"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(os.getenv('CHFS_BLOCK_SIZE', io.DEFAULT_BUFFER_SIZE))

    def __del__(self):
        pass

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if not self.exists(path):
            fd = pychfs.create(path, os.O_RDWR, stat.S_IRWXU)
        else:
            fd = pychfs.open(path, os.O_RDWR)
        logger.debug('chfs.open fd=%s', fd)
        if block_size is None:
            block_size = self.block_size
        return CHFSFile(fd, block_size, **kwargs)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        if recursive:
            raise NotImplementedError
        if isinstance(path, str):
            path = [path]
        ret = dict()
        for p in path:
            fd = pychfs.open(p, os.O_RDWR)
            b = bytearray()
            while True:
                data = bytearray(b'\x00') * self.block_size
                s = pychfs.read(fd, data, self.block_size)
                if s < 0:
                    raise RuntimeError("pychfs.read failed")
                b += data[:s]
                if s < self.block_size:
                    break
            logger.debug("chfs.cat path=%s fd=%d size=%d b=%s", p, fd, len(b), b)
            pychfs.close(fd)
            ret[p] = b
        if len(ret) == 1:
            return ret[path[0]]
        return ret

    def exists(self, path, **kwargs):
        logger.debug("chfs.exists path=%s", path)
        try:
            self.info(path)
        except ValueError:
            return False
        return True

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        logger.debug("!! NOT IMPLEMENTED [ chfs.find ] !!")
        pass

    def _get_stat_type(self, st_mode):
        if stat.S_ISDIR(st_mode):
            return "directory"
        if stat.S_ISREG(st_mode):
            return "file"
        return "other"

    def info(self, path, **kwargs):
        st = pychfs.stat(path)
        logger.debug("chfs.info path=%s, st=%s", path, st)
        ret = {
            'name': path,
            'size': st["st_size"],
            'type': self._get_stat_type(st["st_mode"]),
            'created': st["st_ctime"],
            'islink': stat.S_ISLNK(st["st_mode"]) != 0,
            'mode': st["st_mode"],
            'uid': st["st_uid"],
            'gid': st["st_gid"],
            'mtime': st["st_mtime"]
        }
        return ret

    def mkdir(self, path, create_parents=True, **kwargs):
        if not create_parents:
            raise NotImplementedError
        pychfs.mkdir(path, stat.S_IRWXU)

    def makedirs(self, path, exist_ok=False):
        if not exist_ok and self.exists(path):
            raise NotImplementedError
        pychfs.mkdir(path, stat.S_IRWXU)

    def pipe(self, path, value=None, **kwargs):
        if isinstance(path, str):
            p = dict()
            p[path] = value
            path = p
        for key in path:
            self.pipe_file(key, path[key])

    def pipe_file(self, path, value, **kwargs):
        if not self.exists(path):
            fd = pychfs.create(path, os.O_RDWR, stat.S_IRWXU)
        else:
            fd = pychfs.open(path, os.O_RDWR)
        size = pychfs.write(fd, value, len(value))
        pychfs.close(fd)
        logger.debug("chfs.pipe_file path=%s, value=%s size=%s", path, repr(value), size)

    def rm(self, path, recursive=False, maxdepth=None):
        logger.debug('chfs.rm path=%s', path)
        pychfs.unlink(path)

    def touch(self, path, truncate=True, **kwargs):
        fd = pychfs.create(path, os.O_RDWR, stat.S_IRWXU)
        logger.debug('chfs.touch fd=%s', fd)

class CHFSClient(CHFSStubClient):
    """FSSpec-compatible wrapper of CHFS.
    """
    protocol = "chfs"

    def __init__(self, server="", **kwargs):
        super().__init__(**kwargs)
        if not server:
            server = os.environ['CHFS_SERVER']
        self.server = server
        logger.debug("CHFSClient is created")
        pychfs.init(server)

    def __del__(self):
        pychfs.term()

class CHFSFile(io.RawIOBase):
    def __init__(
        self, fd, block_size, **kwargs
    ):
        self.fd = fd
        self.block_size = block_size
        self.chfs_closed = False

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return True

    def close(self):
        if self.chfs_closed:
            return
        self.chfs_closed = True
        logger.debug('CHFSFile close fd=%s', self.fd)
        return pychfs.close(self.fd)

    def read(self, size = -1):
        logger.debug('CHFSFile read fd=%s size=%d', self.fd, size)
        if size >= 0:
            b = bytearray(b'\x00') * size
            s = pychfs.read(self.fd, b, size)
            if s < 0:
                raise RuntimeError("pychfs.read failed")
            return b[:s]
        b = bytearray()
        while True:
            data = bytearray(b'\x00') * self.block_size
            s = pychfs.read(self.fd, data, self.block_size)
            logger.debug('CHFSFile read req=%d actual=%d', self.block_size, s)
            if s < 0:
                raise RuntimeError("pychfs.read failed")
            b += data[:s]
            if s < self.block_size:
                break
        return b

    def readinto(self, buf):
        s = pychfs.read(self.fd, buf, len(buf))
        if s < 0:
            raise RuntimeError("pychfs.read failed")
        return s

    def write(self, value):
        logger.debug('CHFSFile write fd=%s value=%s', self.fd, value)
        s = pychfs.write(self.fd, value, len(value))
        if s < 0:
            raise RuntimeError("pychfs.write failed")
        return s

    def __enter__(self):
        self._incontext = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._incontext = False
        self.close()

    def seek(self, offset, whence = os.SEEK_SET):
        logger.debug('CHFSFile seek fd=%s offset=%d whence=%d', self.fd, offset, whence)
        s = pychfs.seek(self.fd, offset, whence)
        if s < 0:
            raise RuntimeError("pychfs.seek failed")
        return s

    def tell(self):
        s = pychfs.seek(self.fd, 0, os.SEEK_CUR)
        if s < 0:
            raise RuntimeError("pychfs.tell failed")
        return s

    def truncate(self, length):
        logger.debug('CHFSFile truncate fd=%s length=%d', self.fd, length)
        s = pychfs.truncate(self.fd, length)
        if s < 0:
            raise RuntimeError("pychfs.truncate failed")
        return s
