import logging

from dask.distributed import Semaphore


logger = logging.getLogger(__name__)


def throttle_method_invocations(m, max_executions, name=None):
    if max_executions > 0:
        logger.info(f'Limit {m} to {max_executions}')
        semaphore_name = name if name is not None else repr(m)
        tasks_semaphore = Semaphore(max_leases=max_executions,
                                    name=semaphore_name)
        return _throttle(m, tasks_semaphore)
    else:
        return m


def _throttle(m, sem):
    def throttled_m(*args, **kwargs):
        with sem:
            logger.info(f'Secured slot to run {m}')
            try:
                return m(*args, **kwargs)
            finally:
                logger.info(f'Release slot used for {m}')

    return throttled_m
