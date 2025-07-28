import unittest

from tico.serialize.circle_graph import CircleModel, CircleSubgraph

from tico.utils.serialize import validate_tensor_shapes


class CircleSerializeTest(unittest.TestCase):
    def test_validate_circle_shape(self):
        mod = CircleModel()
        g = CircleSubgraph(mod)
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        validate_tensor_shapes(g)

    def test_validate_tensor_shape_neg(self):
        mod = CircleModel()
        g = CircleSubgraph(mod)
        g.add_tensor_from_scratch(
            prefix="tensor0",
            shape=[1, 2, 3],
            shape_signature=[-1, 0, 0],  # Invalid shape pair
            dtype=0,
        )
        g.add_tensor_from_scratch(
            prefix="tensor1", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        with self.assertRaises(ValueError):
            validate_tensor_shapes(g)
