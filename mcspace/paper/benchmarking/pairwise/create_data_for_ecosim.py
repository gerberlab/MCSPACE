from pathlib import Path
import numpy as np
import pandas as pd
from mcspace.utils import pickle_load


def combine_samples(reads):
    times = list(reads.keys())
    subjs = list(reads[times[0]].keys())
    combreads = []
    for t in times:
        for s in subjs:
            combreads.append(reads[t][s])
    allreads = np.concatenate(combreads, axis=0)    
    return allreads


def create_case(datapath, outpath, case, threshold = 0.005):
    datafile = datapath / f"data_{case}.pkl"
    alldat = pickle_load(datafile)
    reads = combine_samples(alldat['reads'])

    #* convert to relative abundance
    relabun = reads/(reads.sum(axis=1, keepdims=True))
    #* binarize data
    databin = (relabun > threshold).astype(int)
    data = pd.DataFrame(data=databin)
    outfile = f"bindata_{case}.csv"
    data.to_csv(outpath / outfile, index=False)
    print(f"done case={case}")


def main():
    rootpath = Path("./")
    basepath = rootpath / "paper_cluster" / "pairwise"

    for base_sample in ['Mouse', 'Human']:
        datapath = rootpath / "paper_cluster" / "semi_synthetic_data" / "semisyn_data" / base_sample
        outpath = basepath / "ecosim_data" / base_sample
        outpath.mkdir(exist_ok=True, parents=True)

        #* cases
        if base_sample == 'Mouse':
            nsubj_cases = [1,3,5,7,10]
        npart_cases = [10000, 5000, 1000, 500, 100]
        nreads_cases = [10000, 5000, 1000, 500, 100]
        nclust_cases = [5, 10, 15, 20, 25]
        pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
        dsets = np.arange(10)

        for ds in dsets:
            for nk in nclust_cases:
                case = f"D{ds}_K{nk}_Pdefault_Rdefault_Gdefault_B{base_sample}_Sdefault"
                create_case(datapath, outpath, case)

            for npart in npart_cases:
                case = f"D{ds}_Kdefault_P{npart}_Rdefault_Gdefault_B{base_sample}_Sdefault"
                create_case(datapath, outpath, case)

            for nreads in nreads_cases:
                case = f"D{ds}_Kdefault_Pdefault_R{nreads}_Gdefault_B{base_sample}_Sdefault"
                create_case(datapath, outpath, case)

            for gweight in pgarb_cases:
                case = f"D{ds}_Kdefault_Pdefault_Rdefault_G{gweight}_B{base_sample}_Sdefault"
                create_case(datapath, outpath, case)

            if base_sample == 'Mouse':
                for nsubj in nsubj_cases:
                    case = f"D{ds}_Kdefault_Pdefault_Rdefault_Gdefault_B{base_sample}_S{nsubj}"
                    create_case(datapath, outpath, case)


if __name__ == "__main__":
    main()
