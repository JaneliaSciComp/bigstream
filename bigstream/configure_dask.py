import yaml

from bigstream.configure_bigstream import (configure_logging,
                                           set_cpu_resources)

from dask.distributed import (Worker)
from distributed.diagnostics.plugin import WorkerPlugin
from flatten_json import flatten


class ConfigureWorkerPlugin(WorkerPlugin):

    def __init__(self, logging_config, verbose,
                 worker_cpus=0, worker_threads_per_cpu=1):
        self.logging_config = logging_config
        self.verbose = verbose
        self.worker_cpus = worker_cpus
        self.worker_threads_per_cpu = worker_threads_per_cpu

    def setup(self, worker: Worker):
        self.logger = configure_logging(self.logging_config, self.verbose)
        set_cpu_resources(self.worker_cpus, threads_per_cpu=self.worker_threads_per_cpu)

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
