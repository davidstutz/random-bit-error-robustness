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
                #print(precision, places)
                tensor = torch.rand(10) * 2 - 1
                #print(tensor)

                quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                #print(quantized_tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                #print(dequantized_tensor1)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.AdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                #print(quantized_tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                #print(dequantized_tensor2)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantizationCompare(self):
        for precision in [8]:
            for i in range(100):
                places = precision//4 - 1
                #print(precision, places)
                tensor = torch.rand(10) * 2 - 1
                #print(tensor)

                quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                #print(quantized_tensor)
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                #print(dequantized_tensor1)
                self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                #print(quantized_tensor)
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                #print(dequantized_tensor2)
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

    def testClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantizationBitErrors(self):
        for precision in [4]:
            for i in range(1):
                places = max(0, precision//4 - 1)
                print(precision, places)
                tensor = torch.rand(1000000) * 2 - 1
                rand = torch.rand(tensor.shape[0], 8)
                #print(tensor)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                perturbed_quantized_tensor = common.torch.int_random_flip(quantized_tensor, 0.2, 0.2, protected_bits=quantization.protected_bits, rand=rand)
                #print(quantized_tensor)
                #print(perturbed_quantized_tensor)
                def error_rate(a, b):
                    return torch.sum(common.torch.int_hamming_distance(a, b)).item() / (precision * a.shape[0])
                print(error_rate(quantized_tensor, perturbed_quantized_tensor))
                dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
                perturbed_dequantized_tensor1 = quantization.dequantize(perturbed_quantized_tensor, decimal_range)
                #print(dequantized_tensor1)
                #print(perturbed_dequantized_tensor1)
                def mean_abs_error(a, b):
                    return torch.mean(torch.abs(a - b)).item()
                print(mean_abs_error(dequantized_tensor1, perturbed_dequantized_tensor1))
                #self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

                quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(max_abs_range=1, precision=precision)
                quantized_tensor, decimal_range = quantization.quantize(tensor)
                perturbed_quantized_tensor = common.torch.int_random_flip(quantized_tensor, 0.2, 0.2, protected_bits=quantization.protected_bits, rand=rand)
                #print(quantized_tensor)
                #print(perturbed_quantized_tensor)
                print(error_rate(quantized_tensor, perturbed_quantized_tensor))
                dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
                perturbed_dequantized_tensor2 = quantization.dequantize(perturbed_quantized_tensor, decimal_range)
                #print(dequantized_tensor2)
                #print(perturbed_dequantized_tensor2)
                print(mean_abs_error(dequantized_tensor2, perturbed_dequantized_tensor2))
                #self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

                #self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def testClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantizationBitErrors84(self):
        for i in range(1):
            precision1 = 8
            precision2 = 4
            print(precision1, precision2)
            tensor = torch.rand(1000000) * 2 - 1
            rand = torch.rand(tensor.shape[0], 8)
            #print(tensor)

            quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(max_abs_range=1, precision=precision1)
            quantized_tensor, decimal_range = quantization.quantize(tensor)
            perturbed_quantized_tensor = common.torch.int_random_flip(quantized_tensor, 0.2, 0.2, protected_bits=quantization.protected_bits, rand=rand)
            #print(quantized_tensor)
            #print(perturbed_quantized_tensor)
            def error_rate(a, b, precision):
                return torch.sum(common.torch.int_hamming_distance(a, b)).item() / (precision * a.shape[0])
            print(error_rate(quantized_tensor, perturbed_quantized_tensor, precision1))
            dequantized_tensor1 = quantization.dequantize(quantized_tensor, decimal_range)
            perturbed_dequantized_tensor1 = quantization.dequantize(perturbed_quantized_tensor, decimal_range)
            #print(dequantized_tensor1)
            #print(perturbed_dequantized_tensor1)
            def mean_abs_error(a, b):
                return torch.mean(torch.abs(a - b)).item()
            print(mean_abs_error(dequantized_tensor1, perturbed_dequantized_tensor1))
            #self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor1)).item(), places=places)

            quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(max_abs_range=1, precision=precision2)
            quantized_tensor, decimal_range = quantization.quantize(tensor)
            perturbed_quantized_tensor = common.torch.int_random_flip(quantized_tensor, 0.2, 0.2, protected_bits=quantization.protected_bits, rand=rand)
            #print(quantized_tensor)
            #print(perturbed_quantized_tensor)
            print(error_rate(quantized_tensor, perturbed_quantized_tensor ,precision2))
            dequantized_tensor2 = quantization.dequantize(quantized_tensor, decimal_range)
            perturbed_dequantized_tensor2 = quantization.dequantize(perturbed_quantized_tensor, decimal_range)
            #print(dequantized_tensor2)
            #print(perturbed_dequantized_tensor2)
            print(mean_abs_error(dequantized_tensor2, perturbed_dequantized_tensor2))
            #self.assertAlmostEqual(0, torch.max(torch.abs(tensor - dequantized_tensor2)).item(), places=places)

            #self.assertAlmostEqual(0, torch.max(torch.abs(dequantized_tensor1 - dequantized_tensor2)).item(), places=places)

    def noTestFixedPointQuantizationInt(self):
        N = 3

        int_tensor = torch.IntTensor(N).random_(0, torch.iinfo(torch.int32).max)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=32, max_abs_range=1)
        # no context required!

        dequantized_tensor = quantization.dequantize(int_tensor, context=None)
        quantized_tensor, _ = quantization.quantize(dequantized_tensor)

        print(int_tensor)
        print(dequantized_tensor)
        print(quantized_tensor)
        # does not work, quantized_tensor will look different from the original int tensor!

    def noTestFixedPointQuantizationFlips(self):
        N = 1
        P = 32
        epsilon = 1
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [6]
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())

    def notestFixedPointQuantizationFlips16(self):
        N = 1
        P = 16
        cuda = True

        original_epsilons = []
        quantized_epsilons = []
        # only works for powers of two! But then at least it works perfectly
        max_abs_range = 2

        for j in range(1000):

            tensor = torch.FloatTensor(1).uniform_(-max_abs_range, max_abs_range)
            #tensor = torch.FloatTensor([-0.1592175960540771484375000])
            reference_tensor = torch.FloatTensor(2).uniform_(0.501, 1)
            reference_tensor[0] *= -1
            mask = torch.BoolTensor(N * P).fill_(False)
            quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=max_abs_range)

            if cuda:
                tensor = tensor.cuda()
                reference_tensor = reference_tensor.cuda()
                mask = mask.cuda()

            quantized_tensor, context = quantization.quantize(tensor)
            # _, context = quantization.quantize(reference_tensor)
            #self.assertAlmostEqual(context['max_val'] + 0.5/(2**P), abs(tensor.item()))

            # 6 is highest bit that causes the problem
            # problem solved by using torch.float64 in dequantize
            l = numpy.random.randint(0, 16)
            indices = numpy.random.choice(numpy.arange(P), size=l, replace=False)
            #indices = [3, 9, 6, 2, 5, 8, 13, 15, 12, 1, 14]
            epsilon = len(indices)
            #print(indices)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True

            quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

            quantized_bits = common.torch.int_bits(quantized_tensor)
            quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

            original_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())
            self.assertEqual(epsilon, original_epsilons[-1])

            # ! 2
            dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
            dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

            torch.set_printoptions(precision=25)
            #print('---')
            #print('value:\t\t\t\t', tensor[0])
            #print('quantized value:\t\t %d' % quantized_tensor[0])
            #print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
            #print('dequantized value:\t\t', dequantized_quantized_tensor[0])

            #print('-')
            max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
            min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
            if cuda:
                max_quantized_tensor = max_quantized_tensor.cuda()
                min_quantized_tensor = min_quantized_tensor.cuda()
            #print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
            #print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

            #print('-')
            #print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
            #print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
            #print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
            quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

            dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
            quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

            #print('-')
            #print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

            # will not work! also see test above
            quantized_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())
            #self.assertEqual(epsilon, quantized_epsilons[-1])

        print(original_epsilons)
        print(quantized_epsilons)
        print(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))
        print(numpy.sum(numpy.abs(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))))
        print(numpy.sum(numpy.array(original_epsilons) - numpy.array(quantized_epsilons)))

    def notestAdaptiveFixedPointQuantizationFlips16(self):
        N = 1
        P = 16
        cuda = True

        original_epsilons = []
        quantized_epsilons = []
        # only works for powers of two! But then at least it works perfectly

        for j in range(1000):

            tensor = torch.FloatTensor(1).uniform_(-1, 1)
            #tensor = torch.FloatTensor([-0.1592175960540771484375000])
            reference_tensor = torch.FloatTensor(2).uniform_(0.501, 1)
            reference_tensor[0] *= -1
            mask = torch.BoolTensor(N * P).fill_(False)
            quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision=P)

            if cuda:
                tensor = tensor.cuda()
                reference_tensor = reference_tensor.cuda()
                mask = mask.cuda()

            quantized_tensor, context = quantization.quantize(tensor)
            # _, context = quantization.quantize(reference_tensor)
            #self.assertAlmostEqual(context['max_val'] + 0.5/(2**P), abs(tensor.item()))

            # 6 is highest bit that causes the problem
            # problem solved by using torch.float64 in dequantize
            l = numpy.random.randint(0, 16)
            indices = numpy.random.choice(numpy.arange(P), size=l, replace=False)
            epsilon = len(indices)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True

            quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

            quantized_bits = common.torch.int_bits(quantized_tensor)
            quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

            original_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())
            self.assertEqual(epsilon, original_epsilons[-1])

            # ! 2
            dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
            dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

            torch.set_printoptions(precision=25)
            #print('---')
            #print('value:\t\t\t\t', tensor[0])
            #print('quantized value:\t\t %d' % quantized_tensor[0])
            #print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
            #print('dequantized value:\t\t', dequantized_quantized_tensor[0])

            #print('-')
            max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
            min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
            if cuda:
                max_quantized_tensor = max_quantized_tensor.cuda()
                min_quantized_tensor = min_quantized_tensor.cuda()
            #print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
            #print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

            #print('-')
            #print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
            #print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
            #print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
            quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

            dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
            quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

            #print('-')
            #print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

            # will not work! also see test above
            quantized_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())
            #self.assertEqual(epsilon, quantized_epsilons[-1])

        print(original_epsilons)
        print(quantized_epsilons)
        print(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))
        print(numpy.sum(numpy.abs(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))))
        print(numpy.sum(numpy.array(original_epsilons) - numpy.array(quantized_epsilons)))

    def notestAlternativeFixedPointQuantizationFlips16(self):
        N = 1
        P = 16
        cuda = True

        original_epsilons = []
        quantized_epsilons = []
        # only works for powers of two! But then at least it works perfectly
        max_abs_range = 2

        for j in range(1000):

            tensor = torch.FloatTensor(1).uniform_(-max_abs_range, max_abs_range)
            #tensor = torch.FloatTensor([-0.1592175960540771484375000])
            reference_tensor = torch.FloatTensor(2).uniform_(0.501, 1)
            reference_tensor[0] *= -1
            mask = torch.BoolTensor(N * P).fill_(False)
            quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=max_abs_range)

            if cuda:
                tensor = tensor.cuda()
                reference_tensor = reference_tensor.cuda()
                mask = mask.cuda()

            quantized_tensor, context = quantization.quantize(tensor)
            # _, context = quantization.quantize(reference_tensor)
            #self.assertAlmostEqual(context['max_val'] + 0.5/(2**P), abs(tensor.item()))

            # 6 is highest bit that causes the problem
            # problem solved by using torch.float64 in dequantize
            l = numpy.random.randint(0, 16)
            indices = numpy.random.choice(numpy.arange(P), size=l, replace=False)
            #indices = [3, 9, 6, 2, 5, 8, 13, 15, 12, 1, 14]
            epsilon = len(indices)
            #print(indices)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True

            quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

            quantized_bits = common.torch.int_bits(quantized_tensor)
            quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

            original_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())
            self.assertEqual(epsilon, original_epsilons[-1])

            # ! 2
            dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
            dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

            torch.set_printoptions(precision=25)
            #print('---')
            #print('value:\t\t\t\t', tensor[0])
            #print('quantized value:\t\t %d' % quantized_tensor[0])
            #print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
            #print('dequantized value:\t\t', dequantized_quantized_tensor[0])

            #print('-')
            max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
            min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
            if cuda:
                max_quantized_tensor = max_quantized_tensor.cuda()
                min_quantized_tensor = min_quantized_tensor.cuda()
            #print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
            #print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

            #print('-')
            #print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
            #print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
            #print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
            quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

            dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
            quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

            #print('-')
            #print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

            # will not work! also see test above
            quantized_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())
            #self.assertEqual(epsilon, quantized_epsilons[-1])

        print(original_epsilons)
        print(quantized_epsilons)
        print(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))
        print(numpy.sum(numpy.abs(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))))
        print(numpy.sum(numpy.array(original_epsilons) - numpy.array(quantized_epsilons)))

    def notestAlternativeFixedPointQuantizationFlips16Large(self):
        N = 1
        P = 16
        cuda = True

        N = 3000000
        max_abs_range = 0.25
        tensor = torch.FloatTensor(N).uniform_(-max_abs_range, max_abs_range)
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=max_abs_range)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)
        # _, context = quantization.quantize(reference_tensor)
        #self.assertAlmostEqual(context['max_val'] + 0.5/(2**P), abs(tensor.item()))

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        l = int(0.001*N*P)
        indices = numpy.random.choice(P*N, size=l, replace=False)
        epsilon = len(indices)
        #print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        original_epsilon = torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item()
        self.assertEqual(epsilon, original_epsilon)

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        #print('---')
        #print('value:\t\t\t\t', tensor[0])
        #print('quantized value:\t\t %d' % quantized_tensor[0])
        #print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        #print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        #print('-')
        max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
        min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
        if cuda:
            max_quantized_tensor = max_quantized_tensor.cuda()
            min_quantized_tensor = min_quantized_tensor.cuda()
        #print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
        #print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

        #print('-')
        #print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        #print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        #print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        #print('-')
        #print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        #print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        #print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        quantized_epsilon = torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item()
        #self.assertEqual(epsilon, quantized_epsilon)

        print(abs(quantized_epsilon - original_epsilon), original_epsilon, quantized_epsilon)

    def notestAdaptiveAlternativeFixedPointQuantizationFlips16(self):
        N = 1
        P = 16
        cuda = True

        original_epsilons = []
        quantized_epsilons = []
        # only works for powers of two! But then at least it works perfectly
        max_abs_range = 2

        for j in range(10000):

            tensor = torch.FloatTensor(1).uniform_(-max_abs_range, max_abs_range)
            #tensor = torch.FloatTensor([-0.1592175960540771484375000])
            reference_tensor = torch.FloatTensor(2).uniform_(0.501, 1)
            reference_tensor[0] *= -1
            mask = torch.BoolTensor(N * P).fill_(False)
            quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision=P)

            if cuda:
                tensor = tensor.cuda()
                reference_tensor = reference_tensor.cuda()
                mask = mask.cuda()

            quantized_tensor, context = quantization.quantize(tensor)
            # _, context = quantization.quantize(reference_tensor)
            #self.assertAlmostEqual(context['max_val'] + 0.5/(2**P), abs(tensor.item()))

            # 6 is highest bit that causes the problem
            # problem solved by using torch.float64 in dequantize
            l = numpy.random.randint(0, 16)
            indices = numpy.random.choice(numpy.arange(P), size=l, replace=False)
            #indices = [3, 9, 6, 2, 5, 8, 13, 15, 12, 1, 14]
            epsilon = len(indices)
            #print(indices)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True

            quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

            quantized_bits = common.torch.int_bits(quantized_tensor)
            quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

            original_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())
            self.assertEqual(epsilon, original_epsilons[-1])

            # ! 2
            dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
            dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

            torch.set_printoptions(precision=25)
            #print('---')
            #print('value:\t\t\t\t', tensor[0])
            #print('quantized value:\t\t %d' % quantized_tensor[0])
            #print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
            #print('dequantized value:\t\t', dequantized_quantized_tensor[0])

            #print('-')
            max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
            min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
            if cuda:
                max_quantized_tensor = max_quantized_tensor.cuda()
                min_quantized_tensor = min_quantized_tensor.cuda()
            #print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
            #print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

            #print('-')
            #print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
            #print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
            #print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
            quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

            dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
            quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

            #print('-')
            #print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
            #print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

            # will not work! also see test above
            quantized_epsilons.append(torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())
            #self.assertEqual(epsilon, quantized_epsilons[-1])

        print(original_epsilons)
        print(quantized_epsilons)
        print(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))
        print(numpy.sum(numpy.abs(numpy.array(original_epsilons) - numpy.array(quantized_epsilons))))
        print(numpy.sum(numpy.array(original_epsilons) - numpy.array(quantized_epsilons)))

    def notestCompareQuantizationStability(self):
        def test(quantization, cuda=False, N=1, P=16):
            tensor = torch.FloatTensor(1).uniform_(-quantization.max_abs_range, quantization.max_abs_range)
            mask = torch.BoolTensor(N * P).fill_(False)

            if cuda:
                tensor = tensor.cuda()
                mask = mask.cuda()

            quantized_tensor, context = quantization.quantize(tensor)

            l = numpy.random.randint(0, 16)
            indices = numpy.random.choice(numpy.arange(P), size=l, replace=False)
            epsilon = len(indices)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True

            quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

            #quantized_bits = common.torch.int_bits(quantized_tensor)
            #quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

            original_epsilon = torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item()

            # ! 2
            dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
            dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

            torch.set_printoptions(precision=25)
            # print('---')
            # print('value:\t\t\t\t', tensor[0])
            # print('quantized value:\t\t %d' % quantized_tensor[0])
            # print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
            # print('dequantized value:\t\t', dequantized_quantized_tensor[0])

            # print('-')
            max_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).max])
            min_quantized_tensor = torch.ShortTensor([torch.iinfo(torch.int16).min])
            if cuda:
                max_quantized_tensor = max_quantized_tensor.cuda()
                min_quantized_tensor = min_quantized_tensor.cuda()
            # print('max dequantized', quantization.dequantize(max_quantized_tensor, context).item(), max_quantized_tensor.item())
            # print('min dequantized', quantization.dequantize(min_quantized_tensor, context).item(), min_quantized_tensor.item())

            # print('-')
            # print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
            # print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
            # print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
            quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

            #dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
            #quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

            # print('-')
            # print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
            # print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
            # print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

            quantized_epsilon = torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item()
            between_epsilon = torch.sum(common.torch.int_hamming_distance(quantized_flipped_tensor, quantized_dequantized_quantized_flipped_tensor)).item()

            return original_epsilon, quantized_epsilon, between_epsilon

        quantizations = []
        quantizations.append(common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=0.25))
        quantizations.append(common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1))

        quantizations.append(common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(precision=16, max_abs_range=0.25))
        quantizations.append(common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(precision=16, max_abs_range=1))

        for quantization in quantizations:
            original_epsilons = []
            quantized_epsilons = []
            between_epsilons = []

            for i in range(10000):
                original_epsilon, quantized_epsilon, between_epsilon = test(quantization, cuda=True)
                original_epsilons.append(original_epsilon)
                quantized_epsilons.append(quantized_epsilon)
                between_epsilons.append(between_epsilon)

            print(quantization.__class__.__name__, quantization.max_abs_range)
            print(numpy.sum(numpy.array(between_epsilons)))
            print(numpy.sum(numpy.array(original_epsilons)))
            print(numpy.sum(numpy.array(quantized_epsilons)))

    def noTestFixedPointQuantizationFlips16(self):
        N = 1
        P = 16
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [0,1,2,3,4,5,6]
        epsilon = len(indices)
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())

    def noTestFixedPointQuantizationFlipsCorrect(self):
        N = 1
        P = 32
        epsilon = 1
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1, correct=True)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [6]
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())

    def noTestFixedPointQuantizationFlipsDouble(self):
        N = 1
        P = 32
        epsilon = 1
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1, double=True)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [6]
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())

    def noTestFixedPointQuantizationFlipsCorrectDouble(self):
        N = 1
        P = 32
        epsilon = 1
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1, double=True, correct=True)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [0]
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())

    def noTestStableFixedPointQuantizationFlips(self):
        N = 1
        P = 32
        epsilon = 1
        cuda = True

        # ! 1
        tensor = torch.FloatTensor([0.5])
        mask = torch.BoolTensor(N * P).fill_(False)
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=P, max_abs_range=1)

        if cuda:
            tensor = tensor.cuda()
            mask = mask.cuda()

        quantized_tensor, context = quantization.quantize(tensor)

        # 6 is highest bit that causes the problem
        # problem solved by using torch.float64 in dequantize
        indices = [6]
        print(indices)
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        quantized_flipped_tensor = common.torch.int_flip(quantized_tensor, mask.view(N, P))

        quantized_bits = common.torch.int_bits(quantized_tensor)
        quantized_flipped_bits = common.torch.int_bits(quantized_flipped_tensor)

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_flipped_tensor)).item())

        # ! 2
        dequantized_quantized_tensor = quantization.dequantize(quantized_tensor, context)
        dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_flipped_tensor, context)

        torch.set_printoptions(precision=25)
        print('value:\t\t\t\t', tensor[0])
        print('quantized value:\t\t %d' % quantized_tensor[0])
        print('quantized bits:\t\t\t', ''.join(map(str, list(quantized_bits[0].int().cpu().numpy()))))
        print('dequantized value:\t\t', dequantized_quantized_tensor[0])

        print('---')
        print('dequantized flipped value:\t', dequantized_quantized_flipped_tensor[0])
        print('quantized flipped value:\t %d' % quantized_flipped_tensor[0])
        print('quantized flipped bits:\t\t %s' % ''.join(map(str, list(quantized_flipped_bits[0].int().cpu().numpy()))))

        quantized_dequantized_quantized_tensor, _ = quantization.quantize(dequantized_quantized_tensor, context)
        quantized_dequantized_quantized_flipped_tensor, _ = quantization.quantize(dequantized_quantized_flipped_tensor, context)

        dequantized_quantized_dequantized_quantized_flipped_tensor = quantization.dequantize(quantized_dequantized_quantized_flipped_tensor, context)
        quantized_dequantized_quantized_flipped_bits = common.torch.int_bits(quantized_dequantized_quantized_flipped_tensor)

        print('---')
        print('dq dequantized flipped value:\t', dequantized_quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped value:\t %d' % quantized_dequantized_quantized_flipped_tensor[0])
        print('qd quantized flipped bits:\t %s' % ''.join(map(str, list(quantized_dequantized_quantized_flipped_bits[0].int().cpu().numpy()))))

        # will not work! also see test above
        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(quantized_tensor, quantized_dequantized_quantized_flipped_tensor)).item())


if __name__ == '__main__':
    unittest.main()
