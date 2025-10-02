from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Iterator, List

import torch
from torch.export import ExportedProgram


@dataclass
class SubgraphIdx:
    idx: int
    name: str  # model-wise, unique name


_subgraph_indices: List[SubgraphIdx] = []


def store_subgraph_indices(ep: ExportedProgram):
    global _subgraph_indices

    for idx, (_, name) in enumerate(
        get_all_graph_modules(ep, subgraph_only=True), start=1
    ):
        _subgraph_indices += [SubgraphIdx(idx=idx, name=name)]


def get_subgraph_indices() -> List[SubgraphIdx]:
    global _subgraph_indices
    return _subgraph_indices


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
