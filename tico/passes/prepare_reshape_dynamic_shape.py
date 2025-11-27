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
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import ReshapeArgs


@trace_graph_diff_on_pass
class PrepareReshapeDynamicShape(PassBase):
    """
    This pass prepares dynamic shape arguments for reshape operations.
    
    For reshape operations with dynamic shapes (containing fx.Node or SymInt),
    this pass converts the shape list into a single 1D tensor by:
    1. Converting scalar Node elements to 1D tensors via slice
    2. Converting int/SymInt elements to constant 1D tensors
    3. Concatenating all elements into a single shape tensor
    
    This simplifies the serialization logic by ensuring reshape always receives
    either a constant shape list or a single shape tensor node.
    
    Example:
        Before: %reshape = call_function[target=torch.ops.aten.reshape.default](
                    args=(%x, [%slice_tensor, -1]))
        After:  %const_neg1 = call_function[target=torch.ops.aten.tensor.default](args=([-1],))
                %cat = call_function[target=torch.ops.aten.cat.default](
                    args=([%slice_tensor, %const_neg1], 0))
                %reshape = call_function[target=torch.ops.aten.reshape.default](
                    args=(%x, %cat))
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            if node.target != torch.ops.aten.reshape.default:
                continue

            # Get the shape argument - need to access it directly since
            # ReshapeArgs expects a list, but we might have a Node
            if len(node.args) < 2:
                continue
            
            size = node.args[1]
            
            # If size is already a single Node (already prepared), skip
            if isinstance(size, torch.fx.Node):
                continue

            # Check if this is a dynamic reshape
            is_dynamic = any(isinstance(s, (torch.SymInt, torch.fx.Node)) for s in size)
            
            if not is_dynamic:
                continue

            args = ReshapeArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            input_node = args.input

            # Build list of 1D tensor nodes for each dimension
            shape_elements = []
            
            with graph.inserting_before(node):
                for s in size:
                    if isinstance(s, torch.fx.Node):
                        # Node is already a tensor, but might be scalar
                        # We need to ensure it's 1D [1] shape
                        # Check if it's already 1D with shape [1]
                        s_meta = s.meta.get("val")
                        if s_meta is not None and len(s_meta.shape) == 1 and s_meta.shape[0] == 1:
                            # Already 1D, use as-is
                            shape_elements.append(s)
                        else:
                            # Need to reshape to [1]
                            reshape_node = create_node(
                                graph,
                                torch.ops.aten.reshape.default,
                                args=(s, [1]),
                            )
                            reshape_node.meta["val"] = torch.zeros(1, dtype=torch.int32)
                            shape_elements.append(reshape_node)
                    
                    elif isinstance(s, (int, torch.SymInt)):
                        # Create a constant 1D tensor using full
                        val = int(s)
                        const_node = create_node(
                            graph,
                            torch.ops.aten.full.default,
                            args=([1], val),
                            kwargs={"dtype": torch.int32},
                        )
                        const_node.meta["val"] = torch.tensor([val], dtype=torch.int32)
                        shape_elements.append(const_node)
                    else:
                        raise RuntimeError(f"Unsupported size element: {s} {type(s)}")

                # Concatenate all shape elements
                cat_node = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=(shape_elements, 0),
                )
                # Set metadata for cat output
                cat_node.meta["val"] = torch.zeros(len(size), dtype=torch.int32)

            # Replace the reshape args with the concatenated tensor
            node.args = (input_node, cat_node)
            modified = True
            
            logger.debug(
                f"Prepared dynamic shape for {node.name}: concatenated {len(shape_elements)} elements"
            )

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)
