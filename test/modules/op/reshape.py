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

import torch
from torch.export import Dim

from test.modules.base import TestModuleBase

from test.utils import tag

# Note. tests that call `aten.reshape` or `torch.reshape` are exporeted to aten graph that has `aten.view` instead of `aten.reshape`.


class ReshapeChannelLastTensor(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.ops.aten.reshape.default(x, (4, 256))
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 16, 8, 8).to(memory_format=torch.channels_last),), {}


class SimpleReshapeFirstDimMinus(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.ops.aten.reshape.default(x, [-1, 10])
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 4, 5),), {}


class SimpleReshapeLastDimMinus(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.ops.aten.reshape.default(x, [5, -1])
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 4, 5),), {}


class ReshapeTorchAPI(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.reshape(x, (2, 20))
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 4, 5),), {}


@tag.use_onert
class ReshapeDynamicShape(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Reshape to (batch, -1) where batch is dynamic
        return x.reshape(x.shape[0], -1)

    def get_example_inputs(self):
        return (torch.randn(4, 4, 4),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=2, max=128)
        dynamic_shapes = {
            "x": {0: batch},
        }
        return dynamic_shapes

