# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest
import torch
import torch.fx
from torch.export import export

from tico.passes.quant_const_prop_pass import QuantConstPropPass
from tico.passes.lower_to_slice import LowerSelectCopyToSlice
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.utils import is_target_node
from tico.passes import ops

import unittest

import torch
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp

class SelectQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 가중치: [Out_CH=1024, In_CH=3, H=3, W=3]
        # Per-channel 양자화 축을 0번(1024)으로 가정
        self.weight = torch.nn.Parameter(torch.randn(1024, 3, 3, 3))

    def forward(self, x):
        # 0번 축에서 0번째 인덱스 선택 -> [3, 3, 3]
        # Lowering 후: slice(dim=0, start=0, end=1) -> reshape([3, 3, 3])
        w = self.weight.select(0, 0)
        return x + w

    def get_example_inputs():
        return (torch.randn(3, 3, 3),)


class SelectQuantModelTest(unittest.TestCase):
    def test_pass(self):
        q_m = SelectQuantModel()
        assert isinstance(q_m, torch.nn.Module)

        q_m = prepare(q_m, PTQConfig())

        # Calibration
        for i in range(10):
            cal_args = (torch.randn(3, 3, 3),)
            q_m(*cal_args)

        # Quantization
        q_m = convert(q_m)

        # 5. Export module
        ep = torch.export.export(q_m, (torch.randn(3, 3, 3),))
        # DecomposeFakeQuantize().call(ep)
        # ConstPropPass().call(ep)
        # # (weight - DQ)
        # self.assertEqual(
        #     num_of_ops(
        #         ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
        #     ),
        #     1,
        # )

        # target_pass = RemoveWeightDequantOp()
        # target_pass.call(ep)
        # self.assertEqual(
        #     num_of_ops(
        #         ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
        #     ),
        #     0,
        # )
    
# class QuantAwareConstPropTest(unittest.TestCase):
#     def test_shape_op_folding_with_quant_param(self):
#         model = SelectQuantModel().eval()
#         example_inputs = (torch.randn(3, 3, 3),)
        
#         # 1. Export 및 초기 상태 확인
#         ep = export(model, example_inputs)
        
#         # 가중치 placeholder에 Mock QuantParam 주입 (Per-channel, axis=0)
#         weight_node = None
#         for node in ep.graph.nodes:
#             if node.op == "placeholder" and "weight" in node.name:
#                 weight_node = node
#                 qp = QuantParam()
#                 qp.scale = [0.01] * 1024
#                 qp.zero_point = [0] * 1024
#                 qp.axis = 0
#                 qp.dtype = torch.int8
#                 node.meta[QPARAM_KEY] = qp
#                 break
        
#         self.assertIsNotNone(weight_node, "Weight placeholder not found")

#         # 2. Lowering Pass 실행 (select -> slice + reshape)
#         lower_pass = LowerSelectCopyToSlice()
#         lower_pass.call(ep)

#         # Lowering 확인: select는 사라지고 slice와 reshape가 생겨야 함
#         self.assertTrue(any(is_target_node(n, torch.ops.aten.slice.Tensor) for n in ep.graph.nodes))
#         self.assertTrue(any(is_target_node(n, torch.ops.aten.reshape.default) for n in ep.graph.nodes))

#         # 3. Quantization-Aware ConstPropPass 실행
#         # (우리가 구현한 0번 축 slice 시 scale도 잘리고, reshape 시 axis 추적하는 로직 검증)
#         const_prop = QuantConstPropPass()
#         const_prop.call(ep)

#         # 4. 검증 (Validation)
#         # 모든 Shape 연산 노드(slice, reshape)가 사라지고 상수로 합쳐졌는지 확인
#         for node in ep.graph.nodes:
#             if node.op == "call_function":
#                 target = node.target
#                 self.assertNotIn(target, [torch.ops.aten.slice.Tensor, torch.ops.aten.reshape.default],
#                                  f"Node {node.name} ({target}) was not folded!")

#         # 최종 가중치 입력을 받는 node(add)의 입력 placeholder가 QuantParam을 유지하는지 확인
#         final_weight_node = None
#         for node in ep.graph.nodes:
#             if node.op == "placeholder" and "_prop_tensor_constant" in node.name:
#                 final_weight_node = node
#                 self.assertIn(QPARAM_KEY, node.meta, "QuantParam lost after constant folding")
                
#                 qp = node.meta[QPARAM_KEY]
#                 # select(0,0)에 의해 1024개였던 scale이 1개로 slice 되었어야 함
#                 self.assertEqual(len(qp.scale), 1)
#                 self.assertEqual(len(qp.zero_point), 1)
#                 # reshape([3,3,3])이 되었으므로 기존 axis 정보는 의미가 없어지거나 재조정됨
#                 # (구현 로직에 따라 axis가 업데이트된 결과 확인)
#                 print(f"Folded Weight Scale Length: {len(qp.scale)}, New Axis: {qp.axis}")

if __name__ == "__main__":
    unittest.main()
