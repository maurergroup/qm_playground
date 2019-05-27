import unittest
from qmp.potential import preset_potentials
import numpy as np


class PresetPotentialTestCase(unittest.TestCase):

    def test_free(self):
        for dimension in range(1, 3):
            with self.subTest(dimension=dimension):
                free = preset_potentials.Free(dimension)
                function = free()
                self.assertEqual(function(10), 0)

    def test_harmonic(self):
        expected_values = [25, 50]
        for i, dimension in enumerate(range(1, 3)):
            with self.subTest(dimension=dimension):
                omega = np.full(dimension, 1)
                minimum = np.full(dimension, 0)
                harmonic = preset_potentials.Harmonic(dimension,
                                                      omega=omega,
                                                      minimum=minimum)
                function = harmonic()

                point = np.full(dimension, 5)
                self.assertEqual(function(*point), expected_values[i])

    def test_wall(self):
        expected_values1 = [0, 0]
        expected_values2 = [1, 1]
        for i, dimension in enumerate(range(1, 3)):
            with self.subTest(dimension=dimension):
                position = np.full(dimension, 0)
                height = 1
                width = np.full(dimension, 1)

                wall = preset_potentials.Wall(dimension,
                                              position=position,
                                              width=width,
                                              height=height)
                function = wall()

                point = np.full(dimension, 5)
                self.assertEqual(function(*point), expected_values1[i])
                point = np.full(dimension, 0)
                self.assertEqual(function(*point), expected_values2[i])

    def test_box(self):
        expected_values1 = [0, 0]
        expected_values2 = [10000, 10000]
        for i, dimension in enumerate(range(1, 3)):
            with self.subTest(dimension=dimension):
                position = np.full(dimension, 0)
                height = 10000
                width = np.full(dimension, 1)

                box = preset_potentials.Box(dimension,
                                            position=position,
                                            width=width,
                                            height=height)
                function = box()

                point = np.full(dimension, 0)
                self.assertEqual(function(*point), expected_values1[i])
                point = np.full(dimension, 5)
                self.assertEqual(function(*point), expected_values2[i])

    def test_double_box(self):
        expected_values1 = [0, 0]
        expected_values2 = [10000, 10000]
        for i, dimension in enumerate(range(1, 3)):
            with self.subTest(dimension=dimension):
                position1 = np.full(dimension, 0)
                position2 = np.full(dimension, 5)
                width1 = np.full(dimension, 0.5)
                width2 = np.full(dimension, 0.5)
                height = 10000

                double_box = preset_potentials.DoubleBox(dimension,
                                                         position1=position1,
                                                         position2=position2,
                                                         width1=width1,
                                                         width2=width2,
                                                         height=height)
                function = double_box()

                point = np.full(dimension, 0)
                self.assertEqual(function(*point), expected_values1[i])
                point = np.full(dimension, 7)
                self.assertEqual(function(*point), expected_values2[i])

    def test_morse(self):
        expected_values1 = [0, 0]
        expected_values2 = [1, 2]
        for i, dimension in enumerate(range(1, 3)):
            with self.subTest(dimension=dimension):
                morse_a = np.full(dimension, 1)
                morse_D = np.full(dimension, 1)
                morse_p = np.full(dimension, 1)
                morse = preset_potentials.Morse(dimension,
                                                morse_a=morse_a,
                                                morse_D=morse_D,
                                                morse_p=morse_p)
                function = morse()

                point = np.full(dimension, 1)
                self.assertEqual(function(*point), expected_values1[i])
                point = np.full(dimension, 1000)
                self.assertEqual(function(*point), expected_values2[i])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(
            PresetPotentialTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
