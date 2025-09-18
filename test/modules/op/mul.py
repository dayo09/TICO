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

from test.modules.base import TestModuleBase


class SimpleMulWithTensor(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x * y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(3, 3)), {}


class SimpleMulWithScalar(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x * y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), 5), {}


class MulWithBuiltinFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x * y
        return z

    def get_example_inputs(self):
        return (torch.ones(1), 2.0), {}


class MulWithBuiltinInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x * y
        return z

    def get_example_inputs(self):
        return (torch.ones(1).to(torch.int64), 2), {}

class MulBroadcast(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x * 0.3535
        return z

    def get_example_inputs(self):
        return (torch.randn(1,12,64,590),), {}
