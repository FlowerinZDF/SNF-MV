"""Smoke tests for basic scaffold imports."""

import unittest


class TestSmokeImports(unittest.TestCase):
    """Ensure minimal modules import successfully."""

    def test_imports(self) -> None:
        from src.datasets import WeiboDataset
        from src.evaluators.metrics import accuracy
        from src.models import FullModel
        from src.utils.io import read_jsonl
        from src.utils.seed import set_seed

        self.assertIsNotNone(WeiboDataset)
        self.assertIsNotNone(FullModel)
        self.assertEqual(accuracy(1, 2), 0.5)
        self.assertTrue(callable(read_jsonl))
        self.assertTrue(callable(set_seed))


if __name__ == "__main__":
    unittest.main()
