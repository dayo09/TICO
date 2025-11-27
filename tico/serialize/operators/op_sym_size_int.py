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

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index

@register_node_visitor
class SymSizeIntVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.sym_size.int,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        # args: (input, dim)
        input_node = node.args[0]
        dim = node.args[1]
        
        # 1. Shape op
        op_index_shape = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes
        )
        
        # Create a temporary tensor for shape output
        # The shape of 'Shape' output is [rank_of_input]
        input_tensor = self.graph.get_tensor(input_node)
        rank = len(input_tensor.shape)
        shape_output_shape = [rank]
        
        shape_output = self.graph.add_tensor_from_scratch(
            prefix=f"{node.name}_shape",
            shape=shape_output_shape,
            shape_signature=None,
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node,
        )
        
        shape_op = create_builtin_operator(
            self.graph, op_index_shape, [input_node], [shape_output]
        )
        shape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ShapeOptions
        shape_op.builtinOptions = circle.ShapeOptions.ShapeOptionsT()
        shape_op.builtinOptions.outType = circle.TensorType.TensorType.INT32
        
        self.graph.add_operator(shape_op)
        
        # Handle negative dim
        if dim < 0:
            dim += rank
            
        # 2. StridedSlice to extract the dimension
        # Input: shape_output
        # Output: node (scalar)
        
        op_index_slice = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.STRIDED_SLICE, self._op_codes
        )
        
        # Create const tensors for begin, end, strides
        dim_i32 = torch.tensor([dim], dtype=torch.int32)
        begin_tensor = self.graph.add_const_tensor(dim_i32)
        end_tensor = self.graph.add_const_tensor(dim_i32 + 1)
        strides_tensor = self.graph.add_const_tensor(torch.tensor([1], dtype=torch.int32))
        
        inputs = [shape_output, begin_tensor, end_tensor, strides_tensor]
        outputs = [node]
        
        slice_op = create_builtin_operator(
            self.graph, op_index_slice, inputs, outputs
        )
        
        slice_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.StridedSliceOptions
        option = circle.StridedSliceOptions.StridedSliceOptionsT()
        option.beginMask = 0
        option.endMask = 0
        option.ellipsisMask = 0
        option.newAxisMask = 0
        option.shrinkAxisMask = 1 # Shrink the 0-th axis to make it scalar
        
        slice_op.builtinOptions = option
        
        return slice_op
