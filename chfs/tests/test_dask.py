from dask.distributed import Client, LocalCluster
import fsspec
from chfs.dask import CHFSClientDaemon

def test_dask():
    cluster = LocalCluster()
    client = Client(cluster)
    plugin = CHFSClientDaemon()
    client.register_worker_plugin(plugin)
    def func(path, data):
        fs = fsspec.filesystem("chfs_stub")
        fs.pipe(path, data)
        return 0
    future = client.submit(func, "/tmp/foo", b'abcde')
    counts = future.result()
    print(counts)
