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

from tico.passes import ops
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import SqueezeArgs, UnSqueezeArgs, ViewArgs


@trace_graph_diff_on_pass
class ConvertLayoutOpToReshape(PassBase):
    """
    This pass converts layout transformation Op to reshape if possible.
    This is helpful for further optimization.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            reshape_node = None
            
            if node.target in ops.aten.view:
                view_args = ViewArgs(*node.args, **node.kwargs)
                # Preserve the original size argument which may contain dynamic shapes
                # (e.g., sym_size.int nodes, SymInt values, or -1 for inferred dimensions)
                with graph.inserting_after(node):
                    reshape_node = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        args=(view_args.input, view_args.size),
                    )
                modified = True
                
            elif node.target in ops.aten.unsqueeze:
                unsqueeze_args = UnSqueezeArgs(*node.args, **node.kwargs)
                # For unsqueeze, we need to construct the output shape dynamically
                # to preserve symbolic dimensions from the input
                input_node = unsqueeze_args.input
                dim = unsqueeze_args.dim
                
                # Get input shape - may contain symbolic dimensions
                input_meta = input_node.meta.get("val")
                if input_meta is not None:
                    input_shape = list(input_meta.shape)
                    # Build output shape by inserting 1 at the specified dimension
                    # Preserve any symbolic dimensions (SymInt) from input
                    output_shape = input_shape[:dim] + [1] + input_shape[dim:]
                    
                    with graph.inserting_after(node):
                        reshape_node = create_node(
                            graph,
                            torch.ops.aten.reshape.default,
                            args=(input_node, output_shape),
                        )
                    modified = True
                
            elif node.target in ops.aten.squeeze:
                squeeze_args = SqueezeArgs(*node.args, **node.kwargs)
                # For squeeze, we need to construct the output shape dynamically
                # to preserve symbolic dimensions from the input
                input_node = squeeze_args.input
                dims = squeeze_args.dims
                
                # Get input shape - may contain symbolic dimensions
                input_meta = input_node.meta.get("val")
                assert input_meta is not None

                input_shape = list(input_meta.shape)
                # Remove specific dimension if it's size 1
                for dim in dims:
                    assert input_shape[dim] == 1
                output_shape = []
                for dim in range(len(input_shape)):
                    if dim not in dims:
                        output_shape.append(input_shape[dim])

                with graph.inserting_after(node):
                    reshape_node = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        args=(input_node, output_shape),
                    )
                modified = True
            
            if reshape_node is not None:
                node.replace_all_uses_with(reshape_node, propagate_meta=True)
                logger.debug(f"{node.name} is replaced with {reshape_node.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
