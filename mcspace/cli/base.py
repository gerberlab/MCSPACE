import argparse


class CLIModule:
    def __init__(self, subcommand, docstring):
        self.subcommand = subcommand
        self.docstring = docstring

    def create_parser(self, parser):
        raise NotImplementedError()
    
    def main(self, args):
        raise NotImplementedError()
    

def dispatch(cli_mapping):
    # cli_mapping is a dict from string to CLIModule
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    for subcommand, cli_module in cli_mapping.items():
        cli_module.create_parser(subparsers.add_parser(subcommand))

    args = parser.parse_args()

    if args.subcommand not in cli_mapping:
        print("Subcommand `{in_cmd}` not found. Supported commands: {cmds}".format(
            in_cmd=args.subcommand,
            cmds=",".join(list(cli_mapping.keys()))
        ))
        exit(1)

    cli_module = cli_mapping[args.subcommand]
    cli_module.main(args)
    