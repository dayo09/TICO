from typing import Iterator

import torch
from torch.export import ExportedProgram


def get_all_graph_modules(
    ep: ExportedProgram, subgraph_only: bool = False
) -> Iterator[tuple[torch.fx.GraphModule, str]]:
    """
    Get all graph modules and its name
    """
    if not subgraph_only:
        yield ep.graph_module, ""  # root has no name

    # yield subgraphs
    for node in ep.graph.nodes:
        if node.op == "get_attr":
            graph_module = getattr(node.graph.owning_module, node.target)

            # TODO: Enable recursion (n-depth)
            if isinstance(graph_module, torch.fx.graph_module.GraphModule):
                assert hasattr(graph_module, "meta")
                yield graph_module, getattr(node, "name")
