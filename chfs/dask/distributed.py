from dask.distributed.diagnostics.plugin import WorkerPlugin
import pychfs
import chfs
import os
import logging

class CHFSClientDaemon(WorkerPlugin):
    def __init__(self, server=""):
        if not server:
            server = os.environ['CHFS_SERVER']
        self.server = server

    def setup(self, worker):
        pychfs.init(self.server)

    def teardown(self, worker):
        pychfs.term()

    def transition(self, key, start, finish, **kwargs):
        pass
