# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import CondArgs
from tico.utils.graph import create_node
from tico.utils.subgraph import get_all_graph_modules, freeze_subgraphs
import operator
from torch.utils import _pytree as pytree

@trace_graph_diff_on_pass
class MapSubgraph(PassBase):
    """
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram, _) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                != torch.ops.higher_order.cond
            ):
                continue
            
            cond_args = CondArgs(*node.args, **node.kwargs)
            true_graph = cond_args.true_graph
            false_graph = cond_args.false_graph
            graph_args = cond_args.cond_args
            
            def _set_meta_val(graph_node, graph_module, graph_args):
                def _get_meta_val(node):
                    assert hasattr(node, 'meta'), f"'node' has no attribute named 'meta' (node: {node})"
                    assert "val" in node.meta, f"val key not in node.meta (node: {node}, meta: {node.meta})"
                    return node.meta["val"]

                args, kwargs = pytree.tree_map_only(
                    torch.fx.Node,
                    _get_meta_val,
                    (graph_args, {}),
                )
                
                new_val = graph_module(*args, **kwargs)  # type: ignore[operator]
                graph_node.meta["val"] = new_val
    
            for graph_module, name in get_all_graph_modules(exported_program, subgraph_only=True):
                if true_graph.name == name:
                    _set_meta_val(true_graph, graph_module, graph_args)
                if false_graph.name == name:
                    _set_meta_val(false_graph, graph_module, graph_args)
            
            assert "val" in true_graph.meta, f"{true_graph} has no node.meta['val']"
            assert "val" in false_graph.meta, f"{false_graph} has no node.meta['val']"
            
            freeze_subgraphs(exported_program)
            with graph.inserting_before(node):
                circle_if = create_node(
                    graph,
                    torch.ops.circle_custom.if_,
                    args=(cond_args.condition, cond_args.true_graph, cond_args.false_graph, cond_args.cond_args),
                    kwargs={},
                    origin=node,
                )
                
            for t, f in zip(true_graph.meta['val'], false_graph.meta['val']):
                assert type(t) == type(f)
                assert t.shape == f.shape, f"{t.shape} != {f.shape}"
                assert t.dtype == f.dtype, f"{t.dtype} != {f.dtype}"
                
            circle_if.meta["val"] = true_graph.meta['val'][0]
            
            # FIX ME UNLESS torch.ops.higher_order.cond generates this pattern
            assert len(node.users) == 1
            getitem_node = list(node.users.items())[0][0]
            assert getitem_node.target == operator.getitem
            getitem_node.replace_all_uses_with(circle_if)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
