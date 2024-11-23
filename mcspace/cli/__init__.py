from .base import dispatch
from .process_data import ProcessDataCLI

def main():
    print("Hello world!!")
    # ================== Mapping of subcommands to cli modules.
    clis = [
        ProcessDataCLI(subcommand="parse")
    ]

    dispatch({
        cli.subcommand: cli for cli in clis
    })
