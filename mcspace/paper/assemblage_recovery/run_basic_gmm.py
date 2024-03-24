import numpy as np
from mcspace.utils import pickle_load, pickle_save, RESULT_FILE, MODEL_FILE
from mcspace.comparators.comparator_models import BasicGaussianMixture
from pathlib import Path
import time 
from mcspace.data_utils import get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset


def run_case(basepath, datapath, case, seed, base_sample):
    np.random.seed(seed)

    outpathbase = basepath / "gmm_basic_runs" / base_sample / case
    outpathbase.mkdir(exist_ok=True, parents=True)

    sim_dataset = pickle_load(datapath / f"data_{case}.pkl")
    counts = sim_dataset['reads']
    reads = counts[0]['s1']

    klist = np.arange(2,51)
    for ncomm in klist:
        print(f"...fitting k = {ncomm}")
        outpath = outpathbase / f"K_{ncomm}_seed_{seed}"
        outpath.mkdir(exist_ok=True, parents=True)

        model = BasicGaussianMixture(ncomm)
        model.fit_model(reads)
        results = model.get_params()
        #* save results
        pickle_save(outpath / RESULT_FILE, results)
        pickle_save(outpath / MODEL_FILE, model)
    print(f"done case: {case}")


def get_cases(npart_cases, nreads_cases, nclust_cases, dsets, base_sample):
    all_cases = []

    for ds in dsets:
        # vary number of clusters
        for nk in nclust_cases:
            case = f"D{ds}_K{nk}_Pdefault_Rdefault_B{base_sample}"
            all_cases.append(case)

        # vary particles
        for npart in npart_cases:
            case = f"D{ds}_Kdefault_P{npart}_Rdefault_B{base_sample}"
            all_cases.append(case)

        # vary reads
        for nreads in nreads_cases:
            case = f"D{ds}_Kdefault_Pdefault_R{nreads}_B{base_sample}"
            all_cases.append(case)
    return all_cases


def main(run_idx):
    st = time.time()

    # seed = 0
    base_sample = 'Mouse' #! to do, loop over...
    rootpath = Path("./")
    basepath = rootpath / "paper" / "assemblage_recovery"
    datapath = rootpath / "paper" / "semi_synthetic" / "semisyn_data" / base_sample

    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    dsets = np.arange(10)

    all_cases = get_cases(npart_cases, nreads_cases, nclust_cases, dsets, base_sample)
    print(len(all_cases), "cases")
    case = all_cases[run_idx]

    print(f"running case: {case}")
    for seed in range(5):
        run_case(basepath, datapath, case, seed, base_sample) 
    print("***DONE***")
    et = time.time()
    elapsed_time = et - st
    print(elapsed_time)


if __name__ == "__main__":
    # import sys
    # run_idx = int(sys.argv[1])
    # print(f"running case ID: {run_idx}")
    # main(run_idx=0)

    for i in range(150):
        main(i)
    print("***ALL DONE***")
