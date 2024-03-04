import numpy as np
from mcspace.utils import pickle_load, pickle_save, down_sample_reads_percentage
from pathlib import Path
import pandas as pd


def generate_folds_and_downsampled(counts, outpath, nfolds=5):
    npartfull, notus = counts.shape
    nfolds = 5
    shufinds = np.arange(npartfull)
    np.random.shuffle(shufinds)
    testsubs = np.array_split(shufinds, nfolds)
    fullinds = np.arange(npartfull)

    for k in range(nfolds):
        testinds = testsubs[k]
        traininds = np.setdiff1d(fullinds, testinds)
        train = counts[traininds,:]
        test = counts[testinds,:]
        downsampled = down_sample_reads_percentage(test, 0.5)

        print(f"FOLD = {k}")
        print(train.shape)
        print(test.shape)

        #* save
        pickle_save(outpath / f"train_F{k}.pkl", train)
        pickle_save(outpath / f"test_F{k}.pkl", test)
        pickle_save(outpath / f"ds0.5_F{k}.pkl", downsampled)


def main():
    rootpath = Path("./")
    basepath = rootpath / "paper" / "cross_validation"

