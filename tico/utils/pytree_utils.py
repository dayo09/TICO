import threading

import torch
from packaging.version import Version

from tico.utils import logging
from tico.utils.installed_packages import is_transformers_installed

__all__ = ["register_dynamic_cache"]

def is_transformers_less_4_50():
    import transformers
    return Version(transformers.__version__) < Version(
        "4.50.0"
    )

def is_torch_greater_or_equal_than_2_6():
    return Version(torch.__version__) >= Version("2.6.0")
    

def register_dynamic_cache():
    if is_transformers_less_4_50():
        PyTreeRegistryHelper().register_dynamic_cache_legacy()
    else:
        PyTreeRegistryHelper().register_dynamic_cache()


class PyTreeRegistryHelper:
    """
    Thread-safe singleton helper class for registering custom PyTree nodes.

    This class provides functionality to register DynamicCache as a PyTree node
    for torch.export compatibility. This registration is only needed for
    transformers versions below 4.50.0.

    Thread Safety:
    - Uses a class-level threading.Lock() to ensure thread-safe singleton instantiation
    - Uses the same lock to protect the registration process from concurrent calls
    """

    _instance = None  # Class variable to hold the singleton instance
    _has_called = False  # Flag to track if registration has been performed
    _lock = threading.Lock()  # Class-level lock for thread-safe operations

    def __init__(self):
        """Private constructor to prevent direct instantiation"""
        pass

    def __new__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation using double-checked locking pattern.

        Returns:
            PyTreeRegistryHelper: The singleton instance of this class
        """
        if not cls._instance:
            with cls._lock:  # Acquire lock for thread-safe instantiation
                if not cls._instance:  # Double-check after acquiring lock
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register_dynamic_cache_legacy(self):
        """
        Registers DynamicCache as a PyTree node for torch.export compatibility.

        This method is thread-safe and idempotent - it will only perform the
        registration once, even if called multiple times from different threads.

        Note:
            This registration is only needed for transformers versions below 4.50.0.

        Raises:
            ImportError: If transformers package is not installed
        """
        with self._lock:  # Acquire lock for thread-safe registration
            if self.__class__._has_called:
                logger = logging.getLogger(__name__)
                logger.debug("register_dynamic_cache already called, skipping")
                return

            self.__class__._has_called = True
            logger = logging.getLogger(__name__)
            logger.info("Registering DynamicCache PyTree node")

        if not is_transformers_installed():
            raise ImportError("transformers package is not installed")

        from transformers.cache_utils import DynamicCache

        def _flatten_dynamic_cache(dynamic_cache: DynamicCache):
            if not isinstance(dynamic_cache, DynamicCache):
                raise RuntimeError(
                    "This pytree flattening function should only be applied to DynamicCache"
                )
            if not is_torch_greater_or_equal_than_2_6:
                logger = logging.getLogger(__name__)
                logger.warning_once(
                    "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
                )
            dictionary = {
                "key_cache": getattr(dynamic_cache, "key_cache"),
                "value_cache": getattr(dynamic_cache, "value_cache"),
            }
            return torch.utils._pytree._dict_flatten(dictionary)

        def _flatten_with_keys_dynamic_cache(dynamic_cache: DynamicCache):
            dictionary = {
                "key_cache": getattr(dynamic_cache, "key_cache"),
                "value_cache": getattr(dynamic_cache, "value_cache"),
            }
            return torch.utils._pytree._dict_flatten_with_keys(dictionary)

        def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
            dictionary = torch.utils._pytree._dict_unflatten(values, context)
            cache = DynamicCache()
            for k, v in dictionary.items():
                setattr(cache, k, v)
            return cache

        def _flatten_dynamic_cache_for_fx(cache, spec):
            dictionary = {
                "key_cache": getattr(cache, "key_cache"),
                "value_cache": getattr(cache, "value_cache"),
            }
            return torch.fx._pytree._dict_flatten_spec(dictionary, spec)

        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            _flatten_dynamic_cache,
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
        )
        # TODO: This won't be needed in torch 2.7+.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache, _flatten_dynamic_cache_for_fx
        )
        
    def register_dynamic_cache(self):

        from transformers.cache_utils import (
            DynamicCache,
            DynamicLayer,
            DynamicSlidingWindowLayer,
        )
        def _get_cache_dict(cache: DynamicCache):
            """Convert cache to dictionary format for pytree operations."""
            if any(not isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)) for layer in cache.layers):
                raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

            if not is_torch_greater_or_equal_than_2_6:
                logging.warning("DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions.")

            return {
                "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
                "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
            }


        def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
            dictionary = torch.utils._pytree._dict_unflatten(values, context)
            cache = DynamicCache()
            # Reconstruct layers from keys and values lists
            key_list = dictionary.get("key_cache", [])
            value_list = dictionary.get("value_cache", [])
            for idx in range(max(len(key_list), len(value_list))):
                key = key_list[idx] if idx < len(key_list) else None
                value = value_list[idx] if idx < len(value_list) else None
                cache.update(key, value, idx)
            return cache
        
        try:
            torch.utils._pytree.register_pytree_node(
                DynamicCache,
                lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
                _unflatten_dynamic_cache,
                serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
                flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                    _get_cache_dict(dynamic_cache)
                ),
            )
            # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
            torch.fx._pytree.register_pytree_flatten_spec(
                DynamicCache,
                lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec),
            )
        # Catching this in case there are multiple runs for some test runs
        except ValueError as e:
            if "already registered as pytree node" not in str(e):
                raise