import time

from dask.distributed import Semaphore


def throttle_method_invocations(m, max_executions, name=None):
    if max_executions > 0:
        print(f'Limit {m} to {max_executions}', flush=True)
        semaphore_name = name if name is not None else repr(m)
        tasks_semaphore = Semaphore(max_leases=max_executions,
                                    name=semaphore_name)
        return _throttle(m, tasks_semaphore)
    else:
        return m


def _throttle(m, sem):
    def throttled_m(*args, **kwargs):
        with sem:
            print(f'{time.ctime(time.time())} Secured slot to run {m}',
                  flush=True)
            try:
                return m(*args, **kwargs)
            finally:
                print(f'{time.ctime(time.time())} Release slot used for {m}',
                      flush=True)

    return throttled_m
