from .base import dispatch
from .process_data import ProcessDataCLI
from .model_inference import RunInferenceCLI
from .visualization import RenderAssemblageCLI, RenderAssemblageProportionsCLI, ExportAssociationsCytoscapeCLI

def main():
    # ================== Mapping of subcommands to cli modules.
    clis = [
        ProcessDataCLI(subcommand="parse"),
        RunInferenceCLI(subcommand="infer"),
        RenderAssemblageCLI(subcommand="render-assemblages"),
        RenderAssemblageProportionsCLI(subcommand="render-assemblage-proportions"),
        ExportAssociationsCytoscapeCLI(subcommand="associations-to-cytoscape")
    ]

    dispatch({
        cli.subcommand: cli for cli in clis
    })
