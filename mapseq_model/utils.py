import os 
import logging 
import sys
import configparser 

def get_logger(file_path, level=logging.INFO, stream=False):
    """ Make python logger """
    logger = logging.getLogger('mcmc_logger')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    
    if stream is True:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger

    
def create_dir(filename):
    if not os.path.exists(filename):
        os.mkdir(filename)
        print("Directory " , filename ,  " Created ")
    else:    
        print("Directory " , filename ,  " already exists")
    return
    