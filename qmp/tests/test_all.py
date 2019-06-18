import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir)

    unittest.TextTestRunner(verbosity=2).run(suite)
