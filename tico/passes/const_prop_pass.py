# Portions of this file are adapted from code originally authored by
# Meta Platforms, Inc. and affiliates, licensed under the BSD-style
# license found in the LICENSE file in the root directory of their source tree.

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

# https://github.com/pytorch/executorch/blob/61ddee5/exir/passes/constant_prop_pass.py

from collections import OrderedDict
from typing import List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec
from torch.utils import _pytree as pytree

from tico.serialize.circle_graph import _PRIMITIVE_TYPES
from tico.utils import logging
from tico.utils.graph import create_input_spec, generate_fqn, get_first_user_input
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.utils import get_fake_mode

from tico.utils.const_prop import get_constant_placeholder_to_tensor_dict, has_constant_data, get_data, propagate_constants, erase_constant_node, create_constant_placeholder, create_input_specs

@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class ConstPropPass(PassBase):
    """
    Performs constant folding and constant propagation.

    NOTE The exported program gurantees that parameters, buffers, and constant tensors are lifted out of the graph as inputs.
    It means that the pass need to update input specs after folding the constant nodes.
    # ref: https://pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature

    [WHAT IT DOES]
    [1] Propagate the constants.
    [2] Get propagated data from constant nodes.
    [3] Create the constant placeholder nodes according to the propagated data.
    [4] Create input specs according to the created placeholders.
    [5] Update the input specs.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        # [1], [2]
        const_node_to_tensor: OrderedDict[
            torch.fx.Node, torch.Tensor
        ] = propagate_constants(exported_program)
        # [3]
        placeholders = create_constant_placeholder(
            const_node_to_tensor, exported_program
        )
        # [4]
        new_name_to_spec = create_input_specs(placeholders)

        # [5]
        # Get existing input specs.
        existing_name_to_spec = {
            s.arg.name: s for s in exported_program.graph_signature.input_specs
        }
        # Add the new constants to existing input specs dict.
        existing_name_to_spec.update(new_name_to_spec)
        # Generate new input spec.
        new_input_specs = []
        for node in exported_program.graph.nodes:
            if node.op != "placeholder":
                continue
            assert node.name in existing_name_to_spec, node.name
            new_input_specs.append(existing_name_to_spec[node.name])
        exported_program.graph_signature.input_specs = new_input_specs

        graph.eliminate_dead_code()
        graph_module.recompile()

        logger.debug("Constant nodes are propagated")
        # Constant folding can be done with only one time run. Let's set `modified` to False.
        modified = False
        return PassResult(modified)
