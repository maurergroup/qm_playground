import unittest
from qmp.potential import presets
import numpy as np


class PresetsTestCase(unittest.TestCase):

    def test_free(self):
        for dimension in range(1, 3):
            with self.subTest(dimension=dimension):
                free = presets.Free(dimension)
                function = free()
                self.assertEqual(function(10), 0)
                x = np.linspace(0, 10)
                np.testing.assert_array_equal(function(x),
                                              np.zeros_like(x))
                xx, yy = np.meshgrid(x, x)
                np.testing.assert_array_equal(function(xx, yy),
                                              np.zeros_like(xx))

    def test_harmonic_one_dimension(self):
        harmonic = presets.Harmonic(1,
                                    omega=[1],
                                    minimum=[0])
        function = harmonic()

        self.assertEqual(function(0), 0)
        self.assertEqual(function(2), 4)
        point = np.array([0, 2, 5])
        np.testing.assert_array_equal(function(point), point**2)

    def test_harmonic_two_dimension(self):
        harmonic = presets.Harmonic(2,
                                    omega=[1, 1],
                                    minimum=[0, 0])
        function = harmonic()

        self.assertEqual(function(0, 0), 0)
        self.assertEqual(function(2, 2), 8)
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        np.testing.assert_array_equal(function(x, y), x**2 + y**2)
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(function(xx, yy), xx**2 + yy**2)

    def test_wall_one_dimension(self):
        wall = presets.Wall(1,
                            position=np.array([0]),
                            width=np.array([1]),
                            height=1)
        function = wall()

        self.assertEqual(function(0), 1)
        self.assertEqual(function(2), 0)
        point = np.array([0, 2])
        np.testing.assert_array_equal(function(point), [1, 0])

    def test_wall_two_dimension(self):
        wall = presets.Wall(2,
                            position=np.array([0, 0]),
                            width=np.array([1, 1]),
                            height=1)
        function = wall()

        np.testing.assert_array_equal(function(0, 0), 1)
        np.testing.assert_array_equal(function(2, 2), 0)
        x = np.linspace(0, 4, 2)
        y = np.linspace(0, 4, 2)
        np.testing.assert_array_equal(function(x, y), [1, 0])
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(function(xx, yy), [[1, 0], [0, 0]])

    def test_box_one_dimension(self):
        box = presets.Box(1,
                          position=np.array([0]),
                          width=np.array([1]),
                          height=10000)
        function = box()

        self.assertEqual(function(0), 0)
        self.assertEqual(function(2), 10000)
        point = np.array([0, 2])
        np.testing.assert_array_equal(function(point), [0, 10000])

    def test_box_two_dimension(self):
        box = presets.Box(2,
                          position=np.array([0, 0]),
                          width=np.array([1, 1]),
                          height=10000)
        function = box()

        np.testing.assert_array_equal(function(0, 0), 0)
        np.testing.assert_array_equal(function(2, 2), 10000)
        x = np.linspace(0, 4, 2)
        y = np.linspace(0, 4, 2)
        np.testing.assert_array_equal(function(x, y), [0, 10000])
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(function(xx, yy), [[0, 10000],
                                                         [10000, 10000]])

    def test_double_box_one_dimension(self):
        double_box = presets.DoubleBox(1,
                                       position1=np.array([0]),
                                       position2=np.array([5]),
                                       width1=np.array([1]),
                                       width2=np.array([1]),
                                       height=10000)
        function = double_box()

        self.assertEqual(function(0), 0)
        self.assertEqual(function(2), 10000)
        self.assertEqual(function(5), 0)
        point = np.array([0, 2, 5])
        np.testing.assert_array_equal(function(point), [0, 10000, 0])

    def test_double_box_two_dimension(self):
        double_box = presets.DoubleBox(2,
                                       position1=np.array([0, 0]),
                                       position2=np.array([5, 5]),
                                       width1=np.array([1, 1]),
                                       width2=np.array([1, 1]),
                                       height=10000)
        function = double_box()

        np.testing.assert_array_equal(function(0, 0), 0)
        np.testing.assert_array_equal(function(2, 2), 10000)
        np.testing.assert_array_equal(function(5, 5), 0)
        x = np.linspace(0, 4, 2)
        y = np.linspace(0, 4, 2)
        np.testing.assert_array_equal(function(x, y), [0, 10000])
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(function(xx, yy), [[0, 10000],
                                                         [10000, 10000]])

    def test_morse_one_dimension(self):
        morse = presets.Morse(1,
                              morse_a=np.array([1]),
                              morse_D=np.array([1]),
                              morse_p=np.array([1]))
        function = morse()

        self.assertEqual(function(1), 0)
        self.assertEqual(function(1000), 1)
        point = np.array([1, 1000])
        np.testing.assert_array_equal(function(point), [0, 1])

    def test_morse_two_dimension(self):
        morse = presets.Morse(2,
                              morse_a=np.array([1, 1]),
                              morse_D=np.array([1, 1]),
                              morse_p=np.array([1, 1]))
        function = morse()

        np.testing.assert_array_equal(function(1, 1), 0)
        np.testing.assert_array_equal(function(1000, 1000), 2)
        x = np.linspace(1, 100, 2)
        y = np.linspace(1, 100, 2)
        np.testing.assert_array_equal(function(x, y), [0, 2])
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_array_equal(function(xx, yy), [[0, 1],
                                                         [1, 2]])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            PresetsTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
