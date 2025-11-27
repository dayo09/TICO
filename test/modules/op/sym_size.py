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


@tag.use_onert
class SymSizeSimple(TestModuleBase):
    """
    Simplest test case for sym_size.int generation.
    Just returns the batch size (first dimension).
    """

    def forward(self, x):
        # Accessing x.shape[0] on a dynamic dimension creates sym_size.int
        return x.shape[0]

    def get_example_inputs(self):
        return (torch.randn(2, 4, 4),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        return {"x": {0: batch}}


@tag.use_onert
@tag.skip(reason="Not yet supported")
class SymSizeInReshape(TestModuleBase):
    """
    Test case using sym_size.int in a reshape operation.
    This is a common pattern in dynamic batch size models.
    """

    def forward(self, x):
        batch_size = x.shape[0]
        # Use the dynamic batch size in reshape
        return x.reshape(batch_size, -1)

    def get_example_inputs(self):
        return (torch.randn(2, 4, 4),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        return {"x": {0: batch}}


@tag.use_onert
class SymSizeMultipleDims(TestModuleBase):
    """
    Test case using multiple dynamic dimensions.
    """

    def forward(self, x):
        h = x.shape[1]
        w = x.shape[2]
        # Reshape using multiple dynamic dimensions
        return x.reshape(-1, h * w)

    def get_example_inputs(self):
        return (torch.randn(2, 4, 4),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        return {"x": {0: batch}}

