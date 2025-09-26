import torch
from torch.export import ExportedProgram
from copy import deepcopy
from typing import Iterator, List, Iterator
from dataclasses import dataclass
@dataclass
class FrozenSubgraph:
    idx: int
    name: str # model-wise, unique name
    frozen_graph_module: torch.fx.GraphModule # copied subgraph

_frozen_subgraphs: List[FrozenSubgraph] = []

def freeze_subgraphs(ep: ExportedProgram):
    """
    Freeze subgraphs to provide shape inference logic of FakeTensor.
    """
    for idx, (graph_module, name) in enumerate(get_all_graph_modules(ep, subgraph_only=True), start = 1):
        global _frozen_subgraphs
        _frozen_subgraphs += [FrozenSubgraph(idx = idx, name = name, frozen_graph_module = deepcopy(graph_module))]

def get_frozen_subgraphs() -> List[FrozenSubgraph]:
    global _frozen_subgraphs
    return _frozen_subgraphs


def get_all_graph_modules(ep: ExportedProgram, subgraph_only: bool = False) -> Iterator[tuple[torch.fx.GraphModule, str]]:
    """
    Get all graph modules and its name
    """
    if not subgraph_only:
        yield ep.graph_module, "" # root has no name
    
    # yield subgraphs
    for node in ep.graph.nodes:
        if node.op == "get_attr":
            graph_module = getattr(node.graph.owning_module, node.target)
            
            # TODO: Enable recursion (n-depth)
            if isinstance(graph_module, torch.fx.graph_module.GraphModule):
                assert hasattr(graph_module, 'meta')
                yield graph_module, getattr(node, 'name')
