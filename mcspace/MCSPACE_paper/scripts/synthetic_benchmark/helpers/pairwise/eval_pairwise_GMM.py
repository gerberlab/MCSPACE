import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as stats
from sklearn.metrics import auc, f1_score
import torch
from pathlib import Path
from mcspace.model import MCSPACE
from mcspace.data_utils import get_data
from mcspace.utils import get_device, pickle_load, pickle_save, MODEL_FILE, get_mcspace_cooccur_prob
import time
from pairwise_utils import get_gt_assoc, calc_auc


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    outpath =  rootpath / "results" / "pairwise" / "GMM_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()

    print(rootpath)
    print(outpath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
