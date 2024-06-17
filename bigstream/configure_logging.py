import logging

from logging.config import fileConfig

log_level=logging.DEBUG

def configure_logging(config_file, verbose):
    global log_level 
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if config_file:
        fileConfig(config_file)
    else:
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler()
                            ])
    logger = logging.getLogger()
    return logger


def get_bigstream_logger(name=None):
    logger = logging.getLogger(name)
    return logger
