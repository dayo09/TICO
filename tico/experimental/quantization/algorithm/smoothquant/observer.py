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

import functools
from typing import Any, Dict, List

import torch


class ChannelwiseMaxActsObserver:
    """
    Observer to calcuate channelwise maximum activation
    """

    def __init__(self, model):
        """
        model
            A torch module whose activations are to be analyzed.
        hooks
            A list to store the hooks which are registered to collect activation statistics.
        max_acts
            A dictionary to store the maximum activation values
        """
        self.model = model
        self.hooks: List[Any] = []
        self.max_acts: Dict[str, torch.Tensor] = {}

    def attach(self):
        """
        Attach hooks to compute the maximum activation values per channel by running the given model
        on a dataset.

        WHAT IT DOES:
            Set hooks to collect activation values at the per-channel level.
            For each channel, it will calculate the maximum observed activation across
                 all processed samples.
        """
        self.model.eval()

        def stat_tensor(name, tensor: torch.Tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            coming_max = torch.max(tensor, dim=0)[0]
            if name in self.max_acts:
                self.max_acts[name] = torch.max(self.max_acts[name], coming_max)
            else:
                self.max_acts[name] = coming_max

        def stat_input_hook(m, input, name):
            if isinstance(input, tuple):
                input = input[0]
            stat_tensor(name, input)

        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(
                    m.register_forward_pre_hook(
                        functools.partial(stat_input_hook, name=name)
                    )
                )

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def get_max_acts(self):
        return self.max_acts
