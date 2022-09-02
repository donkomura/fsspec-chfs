# fsspec-chfs

fsspec-chfs is a file system interface to CHFS.
CHFS is a parallel consistent hashing file system created instantly using node-local storages such as persistent memory and NVMe SSD for high performance computing.
This repository includes the integration for Dask.

# Requirements

* [CHFS](https://github.com/otatebe/chfs)
* [PyCHFS](https://github.com/donkomura/PyCHFS.git)

# Getting Started

## Installation

```
$ pip install fsspec-chfs
```

## Create file system

You can create CHFS by `chfsctl` and set `CHFS_SERVER` environmental variable.

```
$ eval `chfsctl start`
$ chlist # show started servers
```

## How to use fsspec-chfs

```python
import fsspec

fs = fsspec.filesystem('chfs')

with fs.open('/hello') as f:
	f.write(b'world')
```

### Use in Dask

fsspec-chfs provides `CHFSClientDaemon` plugin for Dask worker, and it optimizes CHFS initialization/termination in Dask.

```python
client = Client(LocalCluster())
plugin = CHFSClientDaemon()
client.register_worker_plugin(plugin)
def func(path, data):
	fs = fsspec.filesystem("chfs_stub")
	fs.pipe(path, data)
	return 0
future = client.submit(func, "/tmp/foo", b'abcde')
counts = future.result()
```

# Developing

## VSCode devcontainer

You can use VSCode devcontainer to develop fsspec-chfs.
The setup steps are follows:

1. Install Docker and Remote-Container extension.
2. Press Ctrl+Shift+P in VSCode.
3. select `Remote-Container: Open the folder in the Container`

## Testing

```
$ eval `chfsctl start` # start the server and set CHFS_SERVER
$ tox
```
