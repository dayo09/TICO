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



import torch
from tico.serialize.quant_param import QPARAM_KEY, QuantParam

def update_quant_param_after_op(node, input_qparam: QuantParam) -> Optional[QuantParam]:
    """
    연산(target)의 특성에 따라 QuantParam의 axis를 업데이트하거나 
    Strided Slice의 경우 scale/zp를 슬라이싱합니다.
    """
    target = node.target
    new_qparam = QuantParam()
    new_qparam.scale = list(input_qparam.scale)
    new_qparam.zero_point = list(input_qparam.zero_point)
    new_qparam.dtype = input_qparam.dtype
    new_qparam.axis = input_qparam.axis

    # 1. Transpose / Permute (축 교환)
    if target in [torch.ops.aten.transpose.int, torch.ops.aten.permute.default]:
        if target == torch.ops.aten.transpose.int:
            dim0, dim1 = node.args[1], node.args[2]
            if new_qparam.axis == dim0: new_qparam.axis = dim1
            elif new_qparam.axis == dim1: new_qparam.axis = dim0
        else: # permute
            dims = node.args[1]
            new_qparam.axis = dims.index(new_qparam.axis)
        return new_qparam

    # 2. Reshape / View (축 인덱스 계산)
    elif target in [torch.ops.aten.reshape.default, torch.ops.aten.view.default]:
        input_shape = node.args[0].meta['val'].shape
        new_shape = node.args[1]
        
        # 채널 축이 독립적으로 유지되는지 확인 (단순 구현: 크기가 유지되는지만 체크)
        # 실제로는 축이 병합되거나 쪼개지면 Abort 해야 함
        channel_dim_size = input_shape[new_qparam.axis]
        try:
            # 새로운 shape에서 기존 채널 사이즈와 동일한 값을 가진 index를 찾음
            # 중복 사이즈가 있을 경우 모호할 수 있으나, 일단 단순 인덱스 추적
            new_qparam.axis = list(new_shape).index(channel_dim_size)
            return new_qparam
        except ValueError:
            return None # 병합/분리 발생 시 폴딩 중단

    # 3. Strided Slice (데이터 슬라이싱에 맞춰 파라미터도 슬라이싱)
    elif target == torch.ops.aten.slice.Tensor:
        dim = node.args[1]
        if dim == new_qparam.axis:
            start, end, step = node.args[2], node.args[3], node.args[4]
            new_qparam.scale = new_qparam.scale[start:end:step]
            new_qparam.zero_point = new_qparam.zero_point[start:end:step]
        return new_qparam

    return None

def propagate_constants_through_shape(exported_program: ExportedProgram) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    const_node_to_tensor = get_constant_placeholder_to_tensor_dict(exported_program)
    graph: torch.fx.Graph = exported_program.graph_module.graph

    for node in graph.nodes:
        if node.op != "call_function":
            continue
            
        # DQ는 건너뛰지만, DQ의 출력(Float)을 입력으로 받는 Shape Op는 처리해야 함
        if not has_constant_data([node.args, node.kwargs], const_node_to_tensor):
            continue

        args_data, kwargs_data = pytree.tree_map(
            lambda x: get_data(x, exported_program, const_node_to_tensor),
            (node.args, node.kwargs),
        )

        try:
            with torch.no_grad():
                prop_constant_tensor = node.target(*args_data, **kwargs_data)
            
            # [핵심 로직] 양자화 정보 전파
            input_node = node.args[0]
            if isinstance(input_node, torch.fx.Node) and QPARAM_KEY in input_node.meta:
                updated_qparam = update_quant_param_after_op(node, input_node.meta[QPARAM_KEY])
                if updated_qparam:
                    node.meta[QPARAM_KEY] = updated_qparam
            
            const_node_to_tensor[node] = prop_constant_tensor
            
        except Exception:
            # Add.Tensor 같은 계산 연산에서 에러 발생 시 안전하게 스킵
            continue

    return const_node_to_tensor


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class QuantConstPropPass(PassBase):
    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        # [1], [2]
        const_node_to_tensor: OrderedDict[
            torch.fx.Node, torch.Tensor
        ] = propagate_constants_through_shape(exported_program)
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
