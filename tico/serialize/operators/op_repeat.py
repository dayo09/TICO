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
from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_shape,
    to_circle_shape,
    circle_legalize_dtype_to,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import InvalidArgumentError, NotYetSupportedError
from tico.utils.validate_args_kwargs import RepeatArgs


@register_node_visitor
class RepeatVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.repeat.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = RepeatArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        repeats = args.repeats

        # Check ranks
        input_tensor = self.graph.get_tensor(input)
        input_rank = len(input_tensor.shape)
        repeats_len = len(repeats)
        
        if input_rank > repeats_len:
             raise RuntimeError(f"Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
             
        # If input rank < repeats length, we need to reshape input to match rank
        tile_input = input
        if input_rank < repeats_len:
             # We need to prepend 1s to input shape
             # If input is static, we can just compute new shape
             # If input is dynamic, we need to construct shape tensor
             
             # Check if input is dynamic
             input_is_dynamic = input_tensor.shapeSignature is not None
             
             if not input_is_dynamic:
                 new_shape = [1] * (repeats_len - input_rank) + input_tensor.shape
                 # Create Reshape op
                 reshape_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.RESHAPE, self._op_codes)
                 reshaped_input = self.graph.add_tensor_from_scratch(
                     prefix=f"{input.name}_reshaped_for_tile",
                     shape=new_shape,
                     shape_signature=None,
                     dtype=input_tensor.type,
                     source_node=input
                 )
                 
                 # Create shape tensor for Reshape (required by Circle?)
                 # Or just use newShape option.
                 # Using newShape option is enough for static.
                 # But let's provide shape tensor for consistency if possible, or just option.
                 # Existing op_reshape uses shape tensor if available.
                 # Here we can just use option for static.
                 # Wait, op_reshape always provides shape tensor input.
                 # Let's provide it.
                 shape_tensor = self.graph.add_const_tensor(circle_legalize_dtype_to(new_shape, dtype=torch.int32))
                 
                 reshape_op = create_builtin_operator(
                     self.graph, reshape_op_idx, [input, shape_tensor], [reshaped_input]
                 )
                 reshape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
                 reshape_op.builtinOptions = circle.ReshapeOptions.ReshapeOptionsT()
                 reshape_op.builtinOptions.newShape = new_shape
                 self.graph.add_operator(reshape_op)
                 tile_input = reshaped_input
             else:
                 # Dynamic input. Construct shape tensor: [1, 1...] + Shape(input)
                 # 1. Shape op
                 shape_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes)
                 input_shape_tensor = self.graph.add_tensor_from_scratch(
                     prefix=f"{input.name}_shape",
                     shape=[input_rank],
                     shape_signature=None,
                     dtype=circle.TensorType.TensorType.INT32,
                     source_node=input
                 )
                 shape_op = create_builtin_operator(
                     self.graph, shape_op_idx, [input], [input_shape_tensor]
                 )
                 shape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ShapeOptions
                 shape_op.builtinOptions = circle.ShapeOptions.ShapeOptionsT()
                 shape_op.builtinOptions.outType = circle.TensorType.TensorType.INT32
                 self.graph.add_operator(shape_op)
                 
                 # 2. Const tensor for prefix 1s
                 prefix_len = repeats_len - input_rank
                 prefix_shape = [1] * prefix_len
                 prefix_tensor = self.graph.add_const_tensor(circle_legalize_dtype_to(prefix_shape, dtype=torch.int32))
                 
                 # 3. Concat
                 concat_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes)
                 new_shape_tensor = self.graph.add_tensor_from_scratch(
                     prefix=f"{input.name}_new_shape",
                     shape=[repeats_len],
                     shape_signature=None,
                     dtype=circle.TensorType.TensorType.INT32,
                     source_node=input
                 )
                 concat_op = create_builtin_operator(
                     self.graph, concat_op_idx, [prefix_tensor, input_shape_tensor], [new_shape_tensor]
                 )
                 concat_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
                 concat_op.builtinOptions = circle.ConcatenationOptions.ConcatenationOptionsT()
                 concat_op.builtinOptions.axis = 0
                 self.graph.add_operator(concat_op)
                 
                 # 4. Reshape
                 reshape_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.RESHAPE, self._op_codes)
                 # Output shape is dynamic (rank is repeats_len)
                 # We can compute shape signature: [1]*prefix + input_sig
                 new_sig = [1]*prefix_len + (input_tensor.shapeSignature if input_tensor.shapeSignature else input_tensor.shape)
                 # Wait, input_tensor.shape might contain 1 for dynamic.
                 # input_tensor.shapeSignature contains -1.
                 
                 reshaped_input = self.graph.add_tensor_from_scratch(
                     prefix=f"{input.name}_reshaped_for_tile",
                     shape=[1]*prefix_len + input_tensor.shape,
                     shape_signature=new_sig,
                     dtype=input_tensor.type,
                     source_node=input
                 )
                 
                 reshape_op = create_builtin_operator(
                     self.graph, reshape_op_idx, [input, new_shape_tensor], [reshaped_input]
                 )
                 reshape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
                 reshape_op.builtinOptions = circle.ReshapeOptions.ReshapeOptionsT()
                 reshape_op.builtinOptions.newShape = [-1] * repeats_len # Dummy for dynamic
                 self.graph.add_operator(reshape_op)
                 tile_input = reshaped_input

        # Construct multiples tensor
        multiples_tensors = []
        for r in repeats:
            if isinstance(r, torch.fx.Node):
                r_tensor = self.graph.get_tensor(r)
                # Cast to INT32 if needed
                if r_tensor.type == circle.TensorType.TensorType.INT64:
                    cast_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes)
                    r_i32 = self.graph.add_tensor_from_scratch(
                        prefix=f"{r.name}_cast_i32",
                        shape=list(r_tensor.shape),
                        shape_signature=list(r_tensor.shapeSignature) if r_tensor.shapeSignature else None,
                        dtype=circle.TensorType.TensorType.INT32,
                        source_node=r
                    )
                    cast_op = create_builtin_operator(
                        self.graph, cast_op_idx, [r], [r_i32]
                    )
                    cast_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.CastOptions
                    cast_op.builtinOptions = circle.CastOptions.CastOptionsT()
                    cast_op.builtinOptions.inDataType = circle.TensorType.TensorType.INT64
                    cast_op.builtinOptions.outDataType = circle.TensorType.TensorType.INT32
                    self.graph.add_operator(cast_op)
                    r_tensor = r_i32
                    r = r_i32

                # Reshape to [1]
                reshape_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.RESHAPE, self._op_codes)
                reshaped_r = self.graph.add_tensor_from_scratch(
                    prefix=f"{r_tensor.name}_reshaped_1d",
                    shape=[1],
                    shape_signature=[1],
                    dtype=circle.TensorType.TensorType.INT32,
                    source_node=None
                )
                shape_1 = self.graph.add_const_tensor([1])
                
                reshape_op = create_builtin_operator(
                    self.graph, reshape_op_idx, [r_tensor], [reshaped_r]
                )
                reshape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
                reshape_op.builtinOptions = circle.ReshapeOptions.ReshapeOptionsT()
                reshape_op.builtinOptions.newShape = [1]
                self.graph.add_operator(reshape_op)
                
                multiples_tensors.append(reshaped_r)
                
            elif isinstance(r, (int, torch.SymInt)):
                val = int(r)
                t_i32_val = circle_legalize_dtype_to([val], dtype=torch.int32)
                t = self.graph.add_const_tensor(t_i32_val)
                multiples_tensors.append(t)
            else:
                raise RuntimeError(f"Unsupported repeat element: {r} {type(r)}")
        
        # Concatenate multiples
        concat_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes)
        multiples_tensor = self.graph.add_tensor_from_scratch(
            prefix=f"{node.name}_multiples",
            shape=[repeats_len],
            shape_signature=None, # multiples is always static shape [rank]
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node
        )
        concat_op = create_builtin_operator(
            self.graph, concat_op_idx, multiples_tensors, [multiples_tensor]
        )
        concat_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
        concat_op.builtinOptions = circle.ConcatenationOptions.ConcatenationOptionsT()
        concat_op.builtinOptions.axis = 0
        self.graph.add_operator(concat_op)
        
        # Tile op
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.TILE, self._op_codes
        )
        inputs = [tile_input, multiples_tensor]
        outputs = [node]
        
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.TileOptions
        operator.builtinOptions = circle.TileOptions.TileOptionsT()
        
        return operator
