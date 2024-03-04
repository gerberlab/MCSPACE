import numpy as np
from mcspace.utils import pickle_load, pickle_save, down_sample_reads_percentage
from pathlib import Path
import pandas as pd
from mcspace.data_utils import get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset


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
    outpathbase = basepath / f"holdout_data"
    outpathbase.mkdir(exist_ok=True, parents=True)

    dsets = [get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset]
    names = ['Human', 'Mouse']

    for dset, name in zip(dsets, names):
        reads, num_otus, times, subjects, dataset = dset()
        for t in times:
            for s in subjects:
                counts = reads[t][s]
                outpath = outpathbase / f"{name}_{t}_{s}"
                outpath.mkdir(exist_ok=True, parents=True)
                generate_folds_and_downsampled(counts, outpath, nfolds=5)
                print(f"DONE: {name}: time = {t}, subject = {s}")
    print("***ALL DONE***")


if __name__ == "__main__":
    main()


