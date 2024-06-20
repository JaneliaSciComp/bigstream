import yaml

from bigstream.configure_logging import configure_logging
from dask.distributed import (Worker)
from distributed.diagnostics.plugin import WorkerPlugin
from flatten_json import flatten


class ConfigureWorkerLoggingPlugin(WorkerPlugin):

    def __init__(self, logging_config, verbose):
        self.logging_config = logging_config
        self.verbose = verbose

    def setup(self, worker: Worker):
        self.logger = configure_logging(self.logging_config, self.verbose)

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        pass

    def release_key(self, key: str, state: str, cause: str | None, reason: None, report: bool):
        pass


def load_dask_config(config_file):
    if (config_file):
        import dask.config
        with open(config_file) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)
