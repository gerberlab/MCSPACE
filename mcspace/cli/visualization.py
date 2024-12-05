from .base import CLIModule
from mcspace.visualization import render_assemblages, render_assemblage_proportions, export_association_networks_to_cytoscape
from mcspace.utils import pickle_load
from pathlib import Path

class RenderAssemblageCLI(CLIModule):
    def __init__(self, subcommand):
        super().__init__(subcommand=subcommand,
                         docstring=__doc__)

    def create_parser(self, parser):
        parser.add_argument('--results','-r',type=str,dest='results_file',required=True,
                            help='Name of inference results pickle file')
        parser.add_argument('--outfile','-o',type=str,dest='outfile',required=True,
                            help='Name of file to save plot to. File extension gives the file format.')
        parser.add_argument('--otu_threshold','-otut',type=float,dest='otu_threshold',
                            required=False,default=0.05,help='Filtering threshold below which to exclude OTUs in plot')
        parser.add_argument('--tree','-t',type=str,dest='treefile',
                            required=False,default=None,help='Name of phylogenetic tree file, if available, on which to plot assemblages')
        parser.add_argument('--fontsize','-f',type=float,dest='fontsize',
                            required=False,default=6,help='Text fontsize for figure')
        parser.add_argument('--legend_off',dest='legend',action='store_false',default=True,
                            help='Option to not include legend in plot')

    def main(self, args):
        results = pickle_load(args.results_file)
        render_assemblages(results,
                           args.outfile,
                           otu_threshold=args.otu_threshold,
                           treefile=Path(args.treefile),
                           fontsize=args.fontsize,
                           legend=args.legend)


class RenderAssemblageProportionsCLI(CLIModule):
    def __init__(self, subcommand):
        super().__init__(subcommand=subcommand,
                         docstring=__doc__)

    def create_parser(self, parser):
        parser.add_argument('--results', '-r', type=str, dest='results_file', required=True,
                            help='Name of inference results pickle file')
        parser.add_argument('--outfile', '-o', type=str, dest='outfile', required=True,
                            help='Name of file to save plot to. File extension gives the file format.')
        parser.add_argument('--average_subjects','-ave',dest='average_subjects',action='store_true',
                            default=False,help='Option to plot subject averaged assemblage proportions')
        parser.add_argument('--annotate_bayes','-bf',dest='annotate_bayes_factors',action='store_true',
                            default=False,help='Option to add Bayes factor annotation')
        parser.add_argument('--logscale_off','-logoff',dest='logscale',action='store_false',
                            default=True,help='Option to plot proportions on linear scale. Be sure to adjust vmin if using.')
        parser.add_argument('--vmin','-vmin',dest='vmin',type=float,required=False,default=-3,
                            help='Minimum value to plot. If using logscale, corresponds to minimum power of ten.')
        parser.add_argument('--fontsize', '-f', type=float, dest='fontsize',
                            required=False, default=6, help='Text fontsize for figure')
        parser.add_argument('--legend_off', dest='legend', action='store_false', default=True,
                            help='Option to not include legend in plot')

    def main(self, args):
        results = pickle_load(args.results_file)
        render_assemblage_proportions(results,
                                      args.outfile,
                                      average_subjects=args.average_subjects,
                                      annotate_bayes_factors=args.annotate_bayes_factors,
                                      logscale=args.logscale,
                                      beta_vmin=args.vmin,
                                      fontsize=args.fontsize,
                                      legend=args.legend)

class ExportAssociationsCytoscapeCLI(CLIModule):
    def __init__(self, subcommand):
        super().__init__(subcommand=subcommand,
                         docstring=__doc__)

    def create_parser(self, parser):
        parser.add_argument('--otu', '-otu', type=str, dest='oidx', required=True,
                            help='Otu index of taxon for which to export associations. Eg `Otu1`')
        parser.add_argument('--results', '-r', type=str, dest='results_file', required=True,
                            help='Name of inference results pickle file')
        parser.add_argument('--outfile', '-o', type=str, dest='outfile', required=True,
                            help='Name of file to export network to.')
        parser.add_argument('--ra_threshold','-rt',type=float,dest='ra_threshold',
                            required=False,default=0.01,help='Relative abundance threshold for which taxa to include')
        parser.add_argument('--edge_threshold','-et',type=float,dest='edge_threshold',
                            required=False,default=0.01,help='Association score threshold for which taxa to include')

    def main(self, args):
        results = pickle_load(args.results_file)
        export_association_networks_to_cytoscape(args.oidx,
                                                 results,
                                                 args.outfile,
                                                 ra_threshold=args.ra_threshold,
                                                 edge_threshold=args.edge_threshold)
