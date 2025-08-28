from mcspace.utils import pickle_save
from mcspace.data_utils import parse
from pathlib import Path
from mcspace.inference import run_inference
import torch
import time


def main(rootdir, outdir):
    st = time.time()
    rootpath = Path(rootdir)
    datapath = rootpath / "datasets" / "human_inulin_perturbed"
    basepath = Path(outdir) / "analysis" / "human_inulin"
    outpath = basepath / "runs"
    outpath.mkdir(exist_ok=True, parents=True)

    device = torch.device("cpu")

    processed_data = parse(datapath/"count_data.csv",
                     datapath/"taxonomy.csv",
                     datapath/"perturbation.csv",
                     subjects_remove=None,
                     times_remove=None,
                     otus_remove=None,
                     num_consistent_subjects=1,
                     min_abundance=0.005,
                     min_reads=250,
                     max_reads=10000,
                     device=device)
    
    run_inference(processed_data,
                outpath,
                device=device)

    # get the execution time
    et = time.time()
    elapsed_time = et - st 
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
