from dask.distributed import Client, LocalCluster
from dask_jobqueue import LSFCluster
import dask.config
from pathlib import Path
import os


class distributedState(object):


    def __init__(self):
        self.client = None
        self.cluster = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.closeClient()

    def modifyConfig(self, options):
        dask.config.set(options)

    def setClient(self, client):
        self.client = client

    def setCluster(self, cluster):
        self.cluster = cluster

    def scaleCluster(self, njobs):
        self.cluster.scale(jobs=njobs)

    def closeClient(self):
        self.scaleCluster(0)
        self.client.close()


    def initializeLSFCluster(self,
        cores=1, memory="16GB", processes=1,
        death_timeout="600s", queue="normal", walltime="1:00",
        ncpus=1, threads_per_worker=2,
        mem=16000, **kwargs
    ):
        """
        Initialize a dask_jobqueue.LSFCluster
        LSFCluster API:
        https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster
        """

        # TODO: add group detection for `project` keyword

        if 1 <= threads_per_worker <= 2*cores:
            tpw = threads_per_worker  # shorthand
            env_extra = [
                f"export NUM_MKL_THREADS={tpw}",
                f"export OPENBLAS_NUM_THREADS={tpw}",
                f"export OPENMP_NUM_THREADS={tpw}",
                f"export OMP_NUM_THREADS={tpw}",
            ]
        else:
            raise ValueError("Maximum of 2 threads per core")

        USER = os.environ["USER"]
        HOME = os.environ["HOME"]

        if "local_directory" not in kwargs:
            kwargs["local_directory"] = f"/scratch/{USER}/"

        if "log_directory" not in kwargs:
            log_dir = f"{HOME}/.dask_distributed/"
            Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        cluster = LSFCluster(
            queue=queue,
            walltime=walltime,
            cores=cores,
            ncpus=ncpus,
            memory=memory,
            env_extra=env_extra,
            death_timeout=death_timeout,
            processes=processes,
            mem=mem,
            **kwargs,
        )
        self.setCluster(cluster)


    def initializeLocalCluster(self, **kwargs):
        """
        Initialize a LocalCluster
        """

        if "host" not in kwargs:
            kwargs["host"] = ""
        cluster = LocalCluster(**kwargs)
        self.setCluster(cluster)


    def initializeClient(self, cluster=None):
        """
        Initialize a dask client
        """

        assert (cluster is not None or self.cluster is not None)
        if cluster is not None:
            self.setCluster(cluster)
        self.setClient(Client(self.cluster))

