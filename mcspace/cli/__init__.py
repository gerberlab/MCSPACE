from .base import dispatch


def main():
    print("Hello world!!")
    # ================== Mapping of subcommands to cli modules.
    clis = [
    ]

    dispatch({
        cli.subcommand: cli for cli in clis
    })
