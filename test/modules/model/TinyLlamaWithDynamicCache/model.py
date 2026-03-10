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

"""E2E test: TinyLlama in decode mode with a DynamicCache.

Scenario
--------
Simulates the token-generation (decode) step where a previously-computed
key/value cache is fed back into the model alongside a single new token.

register_dynamic_cache() selects the correct pytree flatten strategy
automatically based on the installed transformers version:

* transformers with DynamicLayer (newer): Layer-based layout (cache.layers)
* transformers without DynamicLayer (e.g. 4.52.x): legacy layout
  (cache.key_cache / cache.value_cache)

register_dynamic_layer() is also called so that if the Layer-based layout is
in use, DynamicLayer objects inside the cache are also pytree-traversable.
It is a safe no-op when DynamicLayer does not exist in the installed
transformers version.
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from tico.utils.pytree_utils import register_dynamic_cache, register_dynamic_layer

from test.modules.base import TestModuleBase

# Number of previously-processed tokens to pre-fill into the cache.
_PAST_SEQ_LEN = 5


class TinyLlamaWithDynamicCache(TestModuleBase):
    """TinyLlama decode step with a pre-populated DynamicCache."""

    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0").to(
            "cpu"
        )
        self.cfg = self.model.config
        self.rtol = 1e-4
        self.atol = 1e-4

        # register_dynamic_cache picks the right flatten strategy for the
        # installed transformers version automatically.
        # register_dynamic_layer is a no-op when DynamicLayer doesn't exist.
        register_dynamic_cache()
        register_dynamic_layer()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        cfg = self.cfg
        num_layers = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )

        # Single new token (decode step).
        input_ids = torch.tensor([[869]], dtype=torch.long)  # token id for '▁.'
        attention_mask = torch.ones(1, _PAST_SEQ_LEN + 1, dtype=torch.long)
        position_ids = torch.tensor([[_PAST_SEQ_LEN]], dtype=torch.long)

        # Build a DynamicCache pre-filled with random past KV pairs.
        past_key_values = DynamicCache()
        for layer_idx in range(num_layers):
            past_key_values.update(
                torch.randn(1, num_kv_heads, _PAST_SEQ_LEN, head_dim),
                torch.randn(1, num_kv_heads, _PAST_SEQ_LEN, head_dim),
                layer_idx,
            )

        return (
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        ), {}
