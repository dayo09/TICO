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

from typing import Dict, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle
from torch._subclasses.fake_tensor import FakeTensor

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import circle_legalize_dtype_to, to_circle_dtype, to_circle_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import SplitWithSizesArgs


@register_node_visitor
class SplitWithSizesVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.split_with_sizes.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SPLIT_V, self._op_codes
        )
        args = SplitWithSizesArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        split_sizes = args.split_sizes
        axis = args.dim

        # Construct split_sizes_tensor
        split_sizes_tensors = []
        for s in split_sizes:
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
                shape_1 = self.graph.add_const_tensor([1])
                
                reshape_op = create_builtin_operator(
                    self.graph, reshape_op_idx, [s_tensor], [reshaped_s]
                )
                reshape_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
                reshape_op.builtinOptions = circle.ReshapeOptions.ReshapeOptionsT()
                reshape_op.builtinOptions.newShape = [1]
                self.graph.add_operator(reshape_op)
                
                split_sizes_tensors.append(reshaped_s)
                
            elif isinstance(s, (int, torch.SymInt)):
                val = int(s)
                t_i32_val = circle_legalize_dtype_to([val], dtype=torch.int32)
                t = self.graph.add_const_tensor(t_i32_val)
                split_sizes_tensors.append(t)
            else:
                raise RuntimeError(f"Unsupported split_size element: {s} {type(s)}")

        # Concatenate split_sizes
        concat_op_idx = get_op_index(circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes)
        split_sizes_tensor = self.graph.add_tensor_from_scratch(
            prefix=f"{node.name}_split_sizes",
            shape=[len(split_sizes)],
            shape_signature=None,
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node
        )
        concat_op = create_builtin_operator(
            self.graph, concat_op_idx, split_sizes_tensors, [split_sizes_tensor]
        )
        concat_op.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
        concat_op.builtinOptions = circle.ConcatenationOptions.ConcatenationOptionsT()
        concat_op.builtinOptions.axis = 0
        self.graph.add_operator(concat_op)

        axis_i32 = circle_legalize_dtype_to(axis, dtype=torch.int32)
        inputs = [input, split_sizes_tensor, axis_i32]

        """
        `split_with_sizes` has multiple output tensors along with `getitem`.
        Unlike other ops, node itself doesn't become a circle tensor. Instead, each `getitem` will be
        a circle tensor.
        
        torch module having `split_with_sizes` may return selected outputs by using `getitem`.
        However, one-compiler assumes that `CircleSplitV` always have all outputs.
        
        So, let's add unused output tensors to compensate this restriction.
        """
        outputs: List[Union[circle.Tensor.TensorT, torch.fx.node.Node]] = []
        sorted_users = sorted(node.users.keys(), key=lambda x: x.args[1])  # type: ignore[arg-type, return-value]
        users_indices = list(usrnode.args[1] for usrnode in sorted_users)
        user_it = iter(sorted_users)
        for idx, _ in enumerate(split_sizes):
            if idx in users_indices:
                user_node = next(user_it)
                outputs.append(user_node)
            else:
                # Let's add unused output tensor to satisfy circle split_v operator scheme
                node_val = node.meta.get("val")
                assert isinstance(node_val, list)
                fake_tensor = node_val[idx]
                assert isinstance(fake_tensor, FakeTensor)
                shape = list(fake_tensor.size())

                c_shape, c_sig = to_circle_shape(shape)

                dtype = to_circle_dtype(fake_tensor.dtype)
                tensor = self.graph.add_tensor_from_scratch(
                    prefix=f"{node.name}_unused_{idx}",
                    shape=c_shape,
                    shape_signature=c_sig,
                    dtype=dtype,
                    source_node=node,
                )
                outputs.append(tensor)

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.SplitVOptions
        option = circle.SplitVOptions.SplitVOptionsT()
        option.numSplits = len(split_sizes)
        operator.builtinOptions = option

        return operator
