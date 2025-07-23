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

from typing import Any, Dict, Type


class TestTag:
    """Central registry for managing test tag"""

    _registry: Dict[Type, Dict[str, Any]] = {}

    @classmethod
    def add(cls, test_class: Type, tag_key: str, tag_value: Any = None) -> None:
        """Add test tag object to a class

        Args:
            test_class: The test class to add tag to
            tag_key: Name of Tag object to add
            tag_value: Tag object to add
        """
        if test_class not in cls._registry:
            cls._registry[test_class] = {}

        cls._registry[test_class][tag_key] = tag_value

    @classmethod
    def has(cls, test_class: Type, tag_key: str) -> bool:
        """Check if a class has specific tag type

        Args:
            test_class: The test class to check
            tag_key: Type of tag object to check for

        Returns:
            bool: True if the tag exists, False otherwise
        """
        return test_class in cls._registry and tag_key in cls._registry[test_class]

    @classmethod
    def get(cls, test_class: Type, tag_key: str, default: Any = None) -> Any:
        """Get tag object for a class

        Args:
            test_class: The test class to get tag from
            tag_key: Type of tag object to retrieve
            default: Default value to return if tag not found

        Returns:
            The tag object or default if not found
        """
        return cls._registry.get(test_class, {}).get(tag_key, default)


####################################################################
##################        Add tag here            ##################
####################################################################


def skip(reason):
    """
    Mark a test class to be skipped with a reason

    e.g.
      @skip(reason="Not implemented yet")
      class MyTest(unittest.TestCase): # <-- This test will be skipped
    """

    def decorator(cls):
        TestTag.add(cls, "skip", {"reason": reason})
        return cls

    return decorator


def skip_if(predicate, reason):
    """Conditionally mark a test class to be skipped with a reason"""
    if predicate:
        return skip(reason)
    return lambda cls: cls


def test_negative(expected_err):
    """Mark a test class as negative test case with expected error"""

    def decorator(cls):
        TestTag.add(cls, "test_negative", {"expected_err": expected_err})
        return cls

    return decorator


def target(cls):
    """Mark a test class as target test case"""
    TestTag.add(cls, "target")
    return cls


def use_onert(cls):
    """Mark a test class to use ONERT runtime"""
    TestTag.add(cls, "use_onert")
    return cls


def test_without_pt2(cls):
    """Mark a test class to not convert along pt2 during test execution"""
    TestTag.add(cls, "test_without_pt2")
    return cls


def test_without_inference(cls):
    """Mark a test class to not run inference during test execution"""
    TestTag.add(cls, "test_without_inference")
    return cls
