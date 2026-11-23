"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from psf_reconstruction_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(len(record["kernel_basis"]) == 3 and all(np.isclose(np.sum(k), 1) for k in record["kernel_basis"]))
        self.assertTrue(np.allclose(np.sum(list(record["calibrations"].values()), axis=-1), 1))
        self.assertTrue(record["frames"][1]["id"] == record["frames"][2]["id"] and record["frames"][1]["calibration"] != record["frames"][2]["calibration"])
        self.assertTrue(record["frames"][1]["invalid"] != record["frames"][2]["invalid"])
