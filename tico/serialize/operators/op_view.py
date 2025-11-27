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

from tico.serialize.circle_graph import CircleSubgraph, is_const
from tico.serialize.circle_mapping import circle_legalize_dtype_to
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import ViewArgs


@register_node_visitor
class ViewVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.view,
        torch.ops.aten.view.default,
        torch.ops.aten.view_copy.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RESHAPE,
            self._op_codes,
        )
        args = ViewArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        size = args.size

        is_dynamic = any(isinstance(s, (torch.SymInt, torch.fx.Node)) for s in size)

        if not is_dynamic:
            assert is_const(size), type(size)

            if isinstance(size, int):
                raise Exception("scalar size conversion is not supported yet.")

            size_i32 = circle_legalize_dtype_to(size, dtype=torch.int32)
            inputs = [input, size_i32]
        else:
            shape_tensors = []
            for s in size:
                if isinstance(s, torch.fx.Node):
                    s_tensor = self.graph.get_tensor(s)
                    
                    # Cast to INT32 if needed
                    if s_tensor.type == circle.TensorType.TensorType.INT64:
                        cast_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes)
                        s_i32 = self.graph.add_tensor_from_scratch(
                            prefix=f"{s.name}_cast_i32",
                            shape=list(s_tensor.shape),
                            shape_signature=list(s_tensor.shapeSignature) if s_tensor.shapeSignature else None,
                            dtype=circle.TensorType.TensorType.INT32,
                            source_node=s
                        )
                        cast_op = create_builtin_operator(
                            self.graph, cast_op_idx, [s], [s_i32]
                        )
                        cast_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.CastOptions
                        cast_op.builtinOptions = circle.CastOptions.CastOptionsT()
                        cast_op.builtinOptions.inDataType = circle.TensorType.TensorType.INT64
                        cast_op.builtinOptions.outDataType = circle.TensorType.TensorType.INT32
                        self.graph.add_operator(cast_op)
                        s_tensor = s_i32
                        s = s_i32
                    
                    # Reshape to [1]
                    reshape_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.RESHAPE, self._op_codes)
                    reshaped_s = self.graph.add_tensor_from_scratch(
                        prefix=f"{s_tensor.name}_reshaped_1d",
                        shape=[1],
                        shape_signature=[1],
                        dtype=circle.TensorType.TensorType.INT32,
                        source_node=None
                    )
                    shape_1_data = circle_legalize_dtype_to([1], dtype=torch.int32)
                    shape_1 = self.graph.add_const_tensor(shape_1_data)
                    
                    reshape_op = create_builtin_operator(
                        self.graph, reshape_op_idx, [s_tensor, shape_1], [reshaped_s]
                    )
                    reshape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
                    reshape_op.builtinOptions = circle.ReshapeOptions.ReshapeOptionsT()
                    reshape_op.builtinOptions.newShape = [1]
                    self.graph.add_operator(reshape_op)
                    
                    shape_tensors.append(reshaped_s)
                    
                elif isinstance(s, (int, torch.SymInt)):
                    val = int(s)
                    t_i32_val = circle_legalize_dtype_to([val], dtype=torch.int32)
                    t = self.graph.add_const_tensor(t_i32_val)
                    shape_tensors.append(t)
                else:
                    raise RuntimeError(f"Unsupported size element: {s} {type(s)}")
            
            # Concatenate
            concat_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes)
            
            shape_tensor_shape = [len(size)]
            shape_tensor = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_shape_tensor",
                shape=shape_tensor_shape,
                shape_signature=shape_tensor_shape,
                dtype=circle.TensorType.TensorType.INT32,
                source_node=node
            )
            
            concat_op = create_builtin_operator(
                self.graph, concat_op_idx, shape_tensors, [shape_tensor]
            )
            concat_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
            concat_op.builtinOptions = circle.ConcatenationOptions.ConcatenationOptionsT()
            concat_op.builtinOptions.axis = 0
            self.graph.add_operator(concat_op)
            
            inputs = [input, shape_tensor]

        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
        )
        option = circle.ReshapeOptions.ReshapeOptionsT()
        if not is_dynamic:
             option.newShape = size_i32.tolist()
        else:
             option.newShape = [-1] * len(size)

        operator.builtinOptions = option

        return operator
