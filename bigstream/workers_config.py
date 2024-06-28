import logging
import os
import yaml

from bigstream.configure_logging import configure_logging
from dask.distributed import (Worker)
from distributed.diagnostics.plugin import WorkerPlugin
from flatten_json import flatten


logger = logging.getLogger(__name__)


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


class SetWorkerEnvironmentPlugin(WorkerPlugin):

    def __init__(self, worker_cpus):
        self.worker_cpus = worker_cpus

    def setup(self, worker: Worker):
        if self.worker_cpus > 0:
            logger.info(f'Set worker {worker.name} cpus: {self.worker_cpus}')
            os.environ['ITK_THREADS'] = f'{self.worker_cpus}'
            os.environ['MKL_NUM_THREADS'] = f'{self.worker_cpus}'
            os.environ['NUM_MKL_THREADS'] = f'{self.worker_cpus}'
            os.environ['OPENBLAS_NUM_THREADS'] = f'{self.worker_cpus}'
            os.environ['OPENMP_NUM_THREADS'] = f'{self.worker_cpus}'
            os.environ['OMP_NUM_THREADS'] = f'{self.worker_cpus}'

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
