import unittest

from tico.serialize.circle_mapping import validate_circle_shape


class CircleSerializeTest(unittest.TestCase):
    def test_validate_circle_shape(self):
        # static shape
        validate_circle_shape(shape=[1, 2, 3], shape_signature=None)
        # dynamic shape
        validate_circle_shape(shape=[1, 2, 3], shape_signature=[1, -1, 3])
        validate_circle_shape(shape=[1, 2, 3], shape_signature=[-1, -1, 3])

        # Invalid dynamic shape
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[1, -1, 2])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[1, -2, 3])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1], shape_signature=[-1, -1])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[])
