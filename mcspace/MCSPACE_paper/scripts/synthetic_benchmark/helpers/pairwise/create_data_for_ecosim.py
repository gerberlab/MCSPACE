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


def main(rootdir, outdir):
    base_sample = 'Human'
    rootpath = Path(rootdir)
    outpathbase = Path(outdir) / "assemblage_recovery"
    datapathbase = Path(outdir) / "semisyn_data" / base_sample

    outpath = outpathbase / "ecosim_data" / base_sample
    outpath.mkdir(exist_ok=True, parents=True)

    #* cases
    npart_cases = [5000, 2500, 1000, 500, 250]
    nreads_cases = [5000, 2500, 1000, 500, 250]
    dsets = np.arange(10)

    for ds in dsets:
        for npart in npart_cases:
            case = f"D{ds}_P{npart}_Rdefault_B{base_sample}"
            create_case(datapathbase, outpath, case)

        for nreads in nreads_cases:
            case = f"D{ds}_Pdefault_R{nreads}_B{base_sample}"
            create_case(datapathbase, outpath, case)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
