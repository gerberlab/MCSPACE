from torch.optim.optimizer import required

from .base import CLIModule
import argparse
from mcspace.data_utils import parse
from mcspace.utils import pickle_save, get_device


class ProcessDataCLI(CLIModule):
    def __init__(self, subcommand):
        super().__init__(subcommand=subcommand,
                         docstring=__doc__)

    def create_parser(self, parser):
        parser.add_argument('--counts', '-c', type=str, dest='counts', required=True,
                            help='Name of file with counts data')
        parser.add_argument('--taxonomy', '-t', type=str, dest='taxonomy', required=True,
                            help='Name of file for table showing taxonomy for each OTU')
        parser.add_argument('--perturbations', '-p', type=str, dest='perturbations', required=True,
                            help='Name of file giving perturbation information')
        parser.add_argument('--outfile', '-o', type=str, dest='outfile', required=True,
                            help='Name of file to save processed data')
        parser.add_argument('--num_consistent_subjects', '-ncs', type=int, dest='num_consistent_subjects',
                            required=False, default=1, help='Number of subjects that should contain each taxon to pass filtering')
        parser.add_argument('--min_reads', '-minr', type=int, dest='min_reads',
                            required=False, default=250, help='Minimum number of reads for each particle for filtering')
        parser.add_argument('--max_reads', '-maxr', type=int, dest='max_reads',
                            required=False, default=10000, help='Maximum number of reads allowed in particles for filtering')
        parser.add_argument('--min_abundance', '-minabun', type=float, dest='min_abundance',
                            required=False, default=0.005, help='Minimum abundance of OTU for filtering')
        parser.add_argument('--device', '-d', type=str, dest='device', choices=['GPU', 'CPU'],
                            required=False, default=None, help='GPU or CPU')

    def main(self, args):
        if args.device is None:
            device = get_device()
        elif args.device == "GPU":
            device = torch.device("cuda:0")
        elif args.device == "CPU":
            device = torch.device("cpu")

        data = parse(args.counts,
              args.taxonomy,
              args.perturbations,
              args.outfile,
              num_consistent_subjects=args.num_consistent_subjects,
              min_reads=args.min_reads,
              max_reads=args.max_reads,
              min_abundance=args.min_abundance,
              device=device)
        pickle_save(args.outfile, data)
