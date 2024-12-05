from .base import CLIModule
import argparse
import torch
from mcspace.utils import get_device, pickle_load
from mcspace.inference import run_inference

class RunInferenceCLI(CLIModule):
    def __init__(self, subcommand):
        super().__init__(subcommand=subcommand,
                         docstring=__doc__)

    def create_parser(self, parser):
        parser.add_argument('--data', '-d', type=str, dest='data', required=True,
                            help='Name of processed data pickle file')
        parser.add_argument('--outdir', '-o', type=str, dest='outdir', required=True,
                            help='Name of directory to which to output results')
        parser.add_argument('--n_seeds', '-nseed', type=int, dest='n_seeds',
                            required=False, default=10,
                            help='Number of seeds (resets) to run inference with')
        parser.add_argument('--n_epochs', '-nepoch', type=int, dest='n_epochs',
                            required=False, default=20000,
                            help='Number of training epochs per model reset')
        parser.add_argument('--learning_rate', '-lr', type=float, dest='learning_rate',
                            required=False, default=5e-3,
                            help='Learning rate parameter')
        parser.add_argument('--num_assemblages', '-k', type=int, dest='num_assemblages',
                            required=False, default=100,
                            help='Total possible number of assemblages in model')
        parser.add_argument('--sparsity_prior', '-sprior', type=float, dest='sparsity_prior',
                            required=False, default=None,
                            help='Prior probability of assemblage being present')
        parser.add_argument('--sparsity_prior_power', '-spower', type=float, dest='sparsity_prior_power',
                            required=False, default=None,
                            help='Power to which to raise sparsity prior to')
        parser.add_argument('--process_variance', '-pvar', type=float, dest='process_variance',
                            required=False, default=0.01,
                            help='Prior location for process variance')
        parser.add_argument('--perturbation_prior', '-pprior', type=float, dest='perturbation_prior',
                            required=False, default=None,
                            help='Prior probability of perturbation effect')

        # parser.add_argument('--feature', action='store_true')
        parser.add_argument('--no_prior_anneal', dest='anneal_prior', action='store_false',
                            default=True)
        parser.add_argument('--no_contamination', dest='use_contamination', action='store_false',
                            default=True)
        parser.add_argument('--no_sparsity', dest='use_sparsity', action='store_false',
                            default=True)
        parser.add_argument('--no_kmeans_init', dest='use_kmeans_init', action='store_false',
                            default=True)

        parser.add_argument('--device', '-dev', type=str, dest='device', choices=['GPU', 'CPU'],
                            required=False, default=None, help='GPU or CPU')

    def main(self, args):
        device = get_device()

        if args.device == "GPU":
            device = torch.device("cuda:0")
        elif args.device == "CPU":
            device = torch.device("cpu")
        data = pickle_load(args.data)

        run_inference(data,
                      args.outdir,
                      n_seeds=args.n_seeds,
                      n_epochs=args.n_epochs,
                      learning_rate=args.learning_rate,
                      num_assemblages=args.num_assemblages,
                      sparsity_prior=args.sparsity_prior,
                      sparsity_power=args.sparsity_prior_power,
                      anneal_prior=args.anneal_prior,
                      process_variance_prior=args.process_variance,
                      perturbation_prior=args.perturbation_prior,
                      use_contamination=args.use_contamination,
                      use_sparsity=args.use_sparsity,
                      use_kmeans_init=args.use_kmeans_init,
                      device=device)
