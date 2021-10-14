import unittest
import numpy
import torch
import math
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.numpy
import common.quantization
import common.torch
torch.set_printoptions(precision=12)


class TestQuantization(unittest.TestCase):
    def checkBits(self, bits, string, precision=32):
        for i in range(precision):
            if string[i] == '1':
                self.assertTrue(bits[i])
            elif string[i] == '0':
                self.assertFalse(bits[i])
            else:
                assert False

    def testAlternativeFixedPointQuantization(self):
        for precision in [8, 16, 32]:
            for range_exponent in range(0, precision // 2 - 1, 2):
                quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=2 ** range_exponent,
                                                                                     precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10) * (2 ** range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testAdaptiveAlternativeFixedPointQuantization(self):
        for precision in [8, 16, 32]:
            for range_exponent in range(0, precision//2 - 1, 2):
                quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testClippedAdaptiveAlternativeFixedPointQuantization(self):
        for precision in [8, 16, 32]:
            for range_exponent in range(0, precision//2 - 1, 2):
                quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(
                    max_abs_range=2 ** range_exponent, precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

        for precision in [8, 16, 32]:
            for range_exponent in range(0, precision//2 - 1, 2):
                max_abs_range = 2 ** (range_exponent - 1)
                quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(
                    max_abs_range=max_abs_range, precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!

                    indices = torch.abs(tensor) <= max_abs_range
                    not_indices = torch.abs(tensor) > max_abs_range
                    if torch.any(not_indices):
                        numpy.testing.assert_almost_equal(2**(range_exponent - 1), torch.abs(dequantized_tensor[not_indices]).numpy())
                    if torch.any(indices):
                        epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                        self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor[indices] - dequantized_tensor[indices])).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testAlternativeUnsignedFixedPointQuantization(self):
        for precision in [8]:
            for range_exponent in range(0, precision // 2 - 1, 2):
                quantization = common.quantization.AlternativeUnsignedFixedPointQuantization(max_abs_range=2 ** range_exponent,
                                                                                     precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10) * (2 ** range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testAdaptiveAlternativeUnsignedFixedPointQuantization(self):
        for precision in [8]:
            for range_exponent in range(0, precision//2 - 1, 2):
                quantization = common.quantization.AdaptiveAlternativeUnsignedFixedPointQuantization(precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testClippedAdaptiveAlternativeUnsignedFixedPointQuantization(self):
        for precision in [8]:
            for range_exponent in range(0, precision//2 - 1, 2):
                quantization = common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization(
                    max_abs_range=2 ** range_exponent, precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!
                    epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                    self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor - dequantized_tensor)).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

        for precision in [8]:
            for range_exponent in range(-1, precision//2 - 1, 2):
                max_abs_range = 2 ** (range_exponent - 1)
                quantization = common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization(
                    max_abs_range=max_abs_range, precision=precision)
                for i in range(1000):
                    tensor = torch.rand(10)*(2**range_exponent) - torch.rand(10)
                    quantized_tensor, decimal_range = quantization.quantize(tensor)
                    dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                    # 3 places due to very large exponents!

                    indices = torch.abs(tensor) <= max_abs_range
                    not_indices = torch.abs(tensor) > max_abs_range
                    if torch.any(not_indices):
                        numpy.testing.assert_almost_equal(2**(range_exponent - 1), torch.abs(dequantized_tensor[not_indices]).numpy())
                    if torch.any(indices):
                        epsilon = 10 ** (-(precision // 4 - 1)) * 2 ** range_exponent
                        self.assertGreaterEqual(epsilon, torch.max(torch.abs(tensor[indices] - dequantized_tensor[indices])).item())

                tensor = torch.zeros(100)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor)).item(), places=7)

    def testAdaptiveAlternativeUnsymmetricFixedPointQuantizationCompare(self):
        for precision in [8, 16, 32]:
            for i in range(100):
                places = min(6, precision//4 - 1)
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.AdaptiveAlternativeUnsymmetricFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testAlternativeUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.AlternativeUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testAdaptiveAlternativeUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.AdaptiveAlternativeUnsignedFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testClippedAdaptiveAlternativeUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.AdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                tensor = torch.rand(10) * 2 - 1

                quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantizationCompare(self):
        for precision in [4, 6, 8]:
            for i in range(1):
                places = precision//4 - 1
                print(precision, places)
                tensor = torch.rand(10) * 2 - 1
                print(tensor)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                print(quantized_tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                print(dequantized_tensor1)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                print(quantized_tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                print(dequantized_tensor2)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)


if __name__ == '__main__':
    unittest.main()
