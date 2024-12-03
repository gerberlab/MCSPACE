from .base import dispatch
from .process_data import ProcessDataCLI
from .model_inference import RunInferenceCLI


def main():
    # ================== Mapping of subcommands to cli modules.
    clis = [
        ProcessDataCLI(subcommand="parse"),
        RunInferenceCLI(subcommand="infer")
    ]

    dispatch({
        cli.subcommand: cli for cli in clis
    })
