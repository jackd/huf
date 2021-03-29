import unittest

from huf import generic_utils


class GenericUtilsTest(unittest.TestCase):
    def test_group_by(self):
        objs = [
            dict(a=0, b=0, c=1),
            dict(a=0, b=0, c=2),
            dict(a=0, b=1, c=2),
        ]
        grouped = generic_utils.group_by(objs, lambda x: (x["a"], x["b"]))
        self.assertEqual(len(grouped), 2)
        self.assertTrue((0, 0) in grouped)
        self.assertEqual(grouped[(0, 0)], objs[:2])
        self.assertEqual(grouped[(0, 1)], objs[2:])


if __name__ == "__main__":
    unittest.main()
