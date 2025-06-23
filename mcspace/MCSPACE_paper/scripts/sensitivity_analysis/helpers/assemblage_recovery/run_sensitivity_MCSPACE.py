import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data
from mcspace.utils import get_device, pickle_load, pickle_save, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
mpl.use('agg')


def run_case(outpathbase, datapathbase, case, prms, seed):
# def run_case(outpathbase, datapathbase, case, seed, base_sample):
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    outpath = outpathbase / "mcspace" / case / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    ds = prms['dataset']
    dsetname = f"data_D{ds}_timeseries.pkl"
    sim_dataset = pickle_load(datapathbase / dsetname)
    reads = sim_dataset['reads']
    times = sim_dataset['times']
    subjects = sim_dataset['subjects']
    perturbed_times = sim_dataset['perturbed_times']
    num_otus = reads[times[0]][subjects[0]].shape[1]
    data = get_data(reads, device)

    num_assemblages = 100

    perturbation_prior = 0.5/num_assemblages
    sparsity_prior = 0.5/num_assemblages

    num_reads = 0
    for t in times:
        for s in subjects:
            num_reads += reads[t][s].sum()
    sparsity_prior_power = 0.005*num_reads

    process_var_prior = 0.01
    add_process_var=True

    model = MCSPACE(num_assemblages,
                    num_otus,
                    times,
                    subjects,
                    perturbed_times,
                    perturbation_prior,
                    sparsity_prior,
                    sparsity_prior_power,
                    process_var_prior,
                    device,
                    add_process_var,
                    use_contamination=True,
                    contamination_clusters=data['group_garbage_clusters'],
                    process_var_prior_scale=prms['process_var_scale'],
                    perturbation_magnitude_prior_scale=prms['perturbation_var_scale'],
                    garbage_prior_scale=prms['garbage_var_scale']
                )
    model.kmeans_init(data, seed=seed)
    model.to(device)

    print('here')

    # TODO: add back when running
    num_epochs = 5 #! FOR NOW, ADD BACK WHEN RUNNGIN 000
    ELBOs = train_model(model, data, num_epochs, anneal_prior=True)
    torch.save(model, outpath / MODEL_FILE)


def get_cases(process_var_scale, perturbation_var_scale, garbage_var_scale, dsets):
    all_cases = []
    run_params = []
    #* only for output naming, running on just 10 datasets and varying model params
    for ds in dsets:
        # vary proces variance param
        for pvar in process_var_scale:
            case = f"D{ds}_PROC{pvar}_PERTdefault_Gdefault"
            all_cases.append(case)
            prms = {'dataset': ds,
                    'process_var_scale': pvar,
                      'perturbation_var_scale': perturbation_var_scale[0],  # default
                      'garbage_var_scale': garbage_var_scale[0]}  # default
            run_params.append(prms)
        # vary perturbation variance param
        for pvar in perturbation_var_scale:
            case = f"D{ds}_PROCdefault_PERT{pvar}_Gdefault"
            all_cases.append(case)
            prms = {'dataset': ds,
                    'process_var_scale': process_var_scale[0],  # default
                      'perturbation_var_scale': pvar,
                      'garbage_var_scale': garbage_var_scale[0]} # default
            run_params.append(prms)
        # vary garbage variance param
        for gvar in garbage_var_scale:
            case = f"D{ds}_PROCdefault_PERTdefault_G{gvar}"
            all_cases.append(case)
            prms = {'dataset': ds,
                    'process_var_scale': process_var_scale[0],  # default
                    'perturbation_var_scale': perturbation_var_scale[0],  # default
                    'garbage_var_scale': gvar}
            run_params.append(prms)
    return all_cases, run_params


def main(rootdir, outdir, run_idx):
    st = time.time()
    n_seeds = 1 # TODO: increase to 10 for final full runs

    rootpath = Path(rootdir)
    outpathbase = Path(outdir) / "sensitivity_analysis" / "assemblage_recovery"
    datapathbase = Path(outdir) / "sensitivity_analysis" / "synthetic"

    process_var_scale = [10, 100, 1000]
    perturbation_var_scale = [100, 1000, 10000]
    garbage_var_scale = [10, 100, 1000]
    dsets = np.arange(10)

    all_cases, run_params = get_cases(
        process_var_scale,
        perturbation_var_scale,
        garbage_var_scale, 
        dsets)
    print(len(all_cases), "cases")
    if len(all_cases) != len(run_params):
        raise ValueError("Number of cases and run parameters do not match.")
    case = all_cases[run_idx]
    prms = run_params[run_idx]

    print(f"running case: {case}")
    for seed in range(n_seeds):
        run_case(outpathbase, datapathbase, case, prms, seed) 
        print("RUNNING...")
    print("***DONE***")
    et = time.time()
    elapsed_time = et - st
    print(elapsed_time)


def run_all(rootdir, outdir):
    ncases = 90 #! 90 total cases
    for ridx in range(ncases):
        main(rootdir, outdir, ridx)


if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", dest='rootpath', help='root path')
    # parser.add_argument("-o", dest='outpath', help='output path')
    # parser.add_argument("-run_all", dest='run_all', help='option to run all cases', action='store_true')    
    # parser.add_argument("-idx", type=int, dest='idx', help='run number (200 total cases for human; 250 for mouse datasets)')
    # parser.add_argument("-dset", dest='dset', help='dataset case (Human or Mouse)')
    # args = parser.parse_args()
    # if args.run_all is True:
    #     print("RUNNING ALL CASES")
    #     run_all(args.rootpath, args.outpath)
    # else:
    #     print(f"Running {args.dset} case {args.idx}")
    #     main(args.rootpath, args.outpath, args.idx, args.dset)
        
    rootdir = "./MCSPACE_paper"
    outdir = "./MCSPACE_paper/output/"

    # main(rootdir, outdir, 89)
    run_all(rootdir, outdir)
