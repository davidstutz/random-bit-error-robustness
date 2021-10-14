import unittest
import numpy
import torch
import sys
import os
import random
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import common.torch


class TestCffi(unittest.TestCase):
    def checkBits(self, bits, string):
        for i in range(bits.size(0)):
            if string[i] == '1':
                self.assertTrue(bits[i])
            elif string[i] == '0':
                self.assertFalse(bits[i])
            else:
                assert False

    def testInt32Bits(self):
        tensor = torch.zeros((1), dtype=torch.int32)
        bits = common.torch.int_bits(tensor)
        assert not torch.any(bits)

        tensor = torch.ones((1), dtype=torch.int32)
        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '10000000000000000000000000000000')

        tensor = torch.ones((1), dtype=torch.int32)*142156
        bits = common.torch.int_bits(tensor)
        #self.checkBits(bits[0], '00000000000000100010101101001100')
        self.checkBits(bits[0], '00110010110101000100000000000000')

        tensor = torch.ones((1), dtype=torch.int32)*4342421
        bits = common.torch.int_bits(tensor)
        #self.checkBits(bits[0], '00000000010000100100001010010101')
        self.checkBits(bits[0], '10101001010000100100001000000000')

    def testInt32MSBProjection(self):
        def test(original, perturbed):
            projected = common.torch.int_msb_projection(original, perturbed)

            original_bits = common.torch.int_bits(original)
            perturbed_bits = common.torch.int_bits(perturbed)
            projected_bits = common.torch.int_bits(projected)

            perturbed_dist = common.torch.int_hamming_distance(original, perturbed)
            projected_dist = common.torch.int_hamming_distance(original, projected)

            self.assertGreaterEqual(1, projected_dist.item())
            if perturbed_dist.item() == 0:
                self.assertEqual(0, projected_dist.item())

            counter = 0
            for i in range(32):
                if original_bits[0, i] != projected_bits[0, i]:
                    self.assertEqual(0, counter)
                    counter += 1

            return projected

        original = torch.IntTensor([15])
        perturbed = torch.IntTensor([7])
        projected = test(original, perturbed)

        original = torch.IntTensor([15])
        perturbed = torch.IntTensor([6])
        projected = test(original, perturbed)

        for i in range(100):
            original = torch.IntTensor([random.randint(-1000000, 1000000)])
            perturbed = common.torch.int_random_flip(original, 0.25, 0.25)
            projected = test(original, perturbed)

    def testInt32HammingDistance(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
            numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((5, 5))
            for i in range(numpy_dist.shape[0]):
                for j in range(numpy_dist.shape[1]):
                    numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)

            numpy.testing.assert_equal(numpy_dist, torch_dist.numpy())

    def testInt32HammingDistanceZero(self):
        for i in range(10):
            numpy_a = numpy.zeros((5, 5)).astype(numpy.int32)
            numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_dist = numpy.zeros((5, 5))
            for i in range(numpy_dist.shape[0]):
                for j in range(numpy_dist.shape[1]):
                    numpy_dist[i, j] = numpy.binary_repr(numpy_b[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)

            numpy.testing.assert_equal(numpy_dist, torch_dist.numpy())

    def testInt32And(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
            numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_and(numpy_a, numpy_b)

            torch_c = common.torch.int_and(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt32Or(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
            numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_or(numpy_a, numpy_b)

            torch_c = common.torch.int_or(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt32Xor(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)
            numpy_b = numpy.random.randint(0, 1000000, size=(5, 5)).astype(numpy.int32)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_xor(numpy_a, numpy_b)

            torch_c = common.torch.int_xor(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt32Flip(self):
        tensor = torch.tensor([0], dtype=torch.int32)
        mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0]]).bool()

        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '00000000000000000000000000000000')
        flipped_tensor = common.torch.int_flip(tensor, mask)
        flipped_bits = common.torch.int_bits(flipped_tensor)
        self.assertEqual(257, flipped_tensor[0])
        self.checkBits(flipped_bits[0], '10000000100000000000000000000000')

    def testInt32Set(self):
        for i in range(32):
            tensor = torch.tensor([0], dtype=torch.int32)
            bits = [0]*32
            bits[i] = 1
            set1 = torch.tensor([bits]).bool()
            set0 = torch.tensor([[0]*32]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '00000000000000000000000000000000')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            if i < 31:
                self.assertEqual(2**i, flipped_tensor[0].item())
            if i == 31:
                self.assertEqual(-2**31, flipped_tensor[0].item())
            self.assertTrue(flipped_bits[0][i])
            flipped_bits[0][i] = False
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 0)

        for i in range(32):
            tensor = torch.tensor([-1], dtype=torch.int32)
            bits = [0] * 32
            bits[i] = 1
            set0 = torch.tensor([bits]).bool()
            set1 = torch.tensor([[0] * 32]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '11111111111111111111111111111111')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            self.assertEqual(flipped_bits[0][i], False)
            flipped_bits[0][i] = True
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 32)

        for i in range(1000):
            tensor = torch.IntTensor(1).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
            indices = numpy.random.choice(32, 2, replace=False)
            bits1 = [0] * 32
            bits1[indices[0]] = 1
            bits0 = [0] * 32
            bits0[indices[1]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(32):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], True)
                elif j == indices[1]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

        for i in range(100):
            tensor = torch.IntTensor(1).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
            indices = numpy.random.choice(32, 1, replace=False)
            bits1 = [0] * 32
            bits1[indices[0]] = 1
            bits0 = [0] * 32
            bits0[indices[0]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(32):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

    def testInt32FlipHammingFixed(self):
        N = 3
        P = 32
        epsilon = 10

        tensor = torch.IntTensor(N).fill_(0)
        bits = common.torch.int_bits(tensor)
        mask = torch.BoolTensor(N * P).fill_(False)

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

        # indices = numpy.random.choice(N*P, size=epsilon, replace=False)
        indices = [69, 22, 93, 95, 26, 29, 77, 44, 57, 14]
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        flipped_bits = common.torch.int_bits(flipped_tensor)
        # print(indices)
        # print(tensor, bits)
        # print(flipped_tensor, flipped_bits)
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt32FlipHamming(self):
        N = 10000
        P = 32
        epsilon = 500

        for i in range(10):
            tensor = torch.IntTensor(N).random_(0, torch.iinfo(torch.int32).max)
            mask = torch.BoolTensor(N * P).fill_(False)

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
            self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

            indices = numpy.random.choice(N * P, size=epsilon, replace=False)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True
            self.assertEqual(epsilon, torch.sum(mask).item())

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))

            self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt32RandomFlip(self):
        protected_bits = [0]*32

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertFalse(numpy.all(close))
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.1, fraction)

    def testInt32RandomFlipNaNInf(self):
        protected_bits = [0]*32

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            self.assertFalse(numpy.any(numpy.isnan(flipped_tensor.cpu().numpy())))
            self.assertFalse(numpy.any(numpy.isinf(flipped_tensor.numpy())))

    def testInt32RandomFlipProtected(self):
        protected_bits = [1]*32

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            numpy.testing.assert_almost_equal(tensor.numpy(), flipped_tensor.numpy())

    def testInt32MaskedRandomFlip(self):
        protected_bits = [0]*32

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))

            flipped_tensor = common.torch.int_masked_random_flip(tensor, mask, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertFalse(numpy.all(close))

            numpy.testing.assert_almost_equal((1 - mask.numpy())*tensor.numpy(), (1 - mask.numpy())*flipped_tensor.numpy())

    def testInt32IndividualRandomFlip(self):
        protected_bits = [0]*32

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).int()

            zero_prob = torch.ones(100, 3, 32, 32, 32)*0.1
            one_prob = torch.ones(100, 3, 32, 32, 32)*0.1

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.1, fraction)

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).int()

            zero_prob = torch.ones(100, 3, 32, 32, 32)*0.5
            one_prob = torch.ones(100, 3, 32, 32, 32)*0

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.5, fraction)

    def testInt32MaskedIndividualRandomFlip(self):
        protected_bits = [0]*32

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.ones(100, 3, 32, 32, 32)
            one_prob = torch.ones(100, 3, 32, 32, 32)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            numpy.testing.assert_equal(numpy.logical_not(close), mask)

            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int32))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.zeros(100, 3, 32, 32, 32)
            one_prob = torch.zeros(100, 3, 32, 32, 32)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertTrue(numpy.all(close))

    def testInt16Bits(self):
        tensor = torch.zeros((1), dtype=torch.int16)
        bits = common.torch.int_bits(tensor)
        assert not torch.any(bits)

        tensor = torch.ones((1), dtype=torch.int16)
        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '1000000000000000')

        tensor = torch.ones((1), dtype=torch.int16)*13012
        bits = common.torch.int_bits(tensor)
        # 0011001011010100
        self.checkBits(bits[0], '0010101101001100')

    def testInt16HammingDistance(self):
        for i in range(10):
            torch_a = torch.ShortTensor(1, 1).random_(0, torch.iinfo(torch.int16).max)
            torch_b = torch.ShortTensor(1, 1).random_(0, torch.iinfo(torch.int16).max)

            numpy_a = torch_a.numpy()
            numpy_b = torch_b.numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
                for j in range(numpy_dist.shape[1]):
                    numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)

            numpy.testing.assert_equal(numpy_dist, torch_dist.numpy())

    def testInt16HammingDistanceZero(self):
        for i in range(10):
            torch_a = torch.ShortTensor(5, 5).fill_(0)
            torch_b = torch.ShortTensor(5, 5).random_(0, torch.iinfo(torch.int16).max)

            numpy_a = torch_a.numpy()
            numpy_b = torch_b.numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
               for j in range(numpy_dist.shape[1]):
                   numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)
            numpy.testing.assert_equal(numpy_dist, torch_dist.numpy())

    def testInt16And(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_and(numpy_a, numpy_b)

            torch_c = common.torch.int_and(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt16Or(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_or(numpy_a, numpy_b)

            torch_c = common.torch.int_or(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt16Xor(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int16).max, size=(5, 5)).astype(numpy.int16)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_xor(numpy_a, numpy_b)

            torch_c = common.torch.int_xor(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.numpy())

    def testInt16Flip(self):
        tensor = torch.tensor([0], dtype=torch.int16)
        mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 0, 0]]).bool()

        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '00000000000000000000000000000000')
        flipped_tensor = common.torch.int_flip(tensor, mask)
        flipped_bits = common.torch.int_bits(flipped_tensor)
        self.assertEqual(257, flipped_tensor[0])
        self.checkBits(flipped_bits[0], '10000000100000000000000000000000')

    def testInt16Set(self):
        for i in range(16):
            tensor = torch.tensor([0], dtype=torch.int16)
            bits = [0]*16
            bits[i] = 1
            set1 = torch.tensor([bits]).bool()
            set0 = torch.tensor([[0]*16]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '0000000000000000')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            if i < 15:
                self.assertEqual(2**i, flipped_tensor[0].item())
            if i == 15:
                self.assertEqual(-2**15, flipped_tensor[0].item())
            self.assertTrue(flipped_bits[0][i])
            flipped_bits[0][i] = False
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 0)

        for i in range(16):
            tensor = torch.tensor([-1], dtype=torch.int16)
            bits = [0] * 16
            bits[i] = 1
            set0 = torch.tensor([bits]).bool()
            set1 = torch.tensor([[0] * 16]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '1111111111111111')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            self.assertEqual(flipped_bits[0][i], False)
            flipped_bits[0][i] = True
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 16)

        for i in range(1000):
            tensor = torch.ShortTensor(1).random_(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
            indices = numpy.random.choice(16, 2, replace=False)
            bits1 = [0] * 16
            bits1[indices[0]] = 1
            bits0 = [0] * 16
            bits0[indices[1]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(16):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], True)
                elif j == indices[1]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

        for i in range(100):
            tensor = torch.ShortTensor(1).random_(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
            indices = numpy.random.choice(16, 1, replace=False)
            bits1 = [0] * 16
            bits1[indices[0]] = 1
            bits0 = [0] * 16
            bits0[indices[0]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(16):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

    def testInt16FlipHammingFixed(self):
        N = 3
        P = 16
        epsilon = 10

        tensor = torch.IntTensor(N).fill_(0).to(torch.int16)
        bits = common.torch.int_bits(tensor)
        mask = torch.BoolTensor(N * P).fill_(False)

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        self.assertEqual(flipped_tensor.dtype, tensor.dtype)
        self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

        indices = numpy.random.choice(N*P, size=epsilon, replace=False)
        #indices = [69, 22, 93, 95, 26, 29, 77, 44, 57, 14]
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        flipped_bits = common.torch.int_bits(flipped_tensor)
        # print(indices)
        # print(tensor, bits)
        # print(flipped_tensor, flipped_bits)
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt16FlipHamming(self):
        N = 10000
        P = 16
        epsilon = 500

        for i in range(10):
            tensor = torch.ShortTensor(N).random_(0, torch.iinfo(torch.int16).max)
            mask = torch.BoolTensor(N * P).fill_(False)

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
            self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

            indices = numpy.random.choice(N * P, size=epsilon, replace=False)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True
            self.assertEqual(epsilon, torch.sum(mask).item())

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))

            self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt16RandomFlip(self):
        protected_bits = [0]*16

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertFalse(numpy.all(close))
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.2, fraction)

    def testInt16RandomFlipNaNInf(self):
        protected_bits = [0]*16

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            self.assertFalse(numpy.any(numpy.isnan(flipped_tensor.cpu().numpy())))
            self.assertFalse(numpy.any(numpy.isinf(flipped_tensor.numpy())))

    def testInt16RandomFlipProtected(self):
        protected_bits = [1]*16

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))
            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)
            numpy.testing.assert_almost_equal(tensor.numpy(), flipped_tensor.numpy())


    def testInt16MaskedRandomFlip(self):
        protected_bits = [0]*16

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))

            flipped_tensor = common.torch.int_masked_random_flip(tensor, mask, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertFalse(numpy.all(close))

            numpy.testing.assert_almost_equal((1 - mask.numpy())*tensor.numpy(), (1 - mask.numpy())*flipped_tensor.numpy())

    def testInt16IndividualRandomFlip(self):
        protected_bits = [0]*16

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.int16)

            zero_prob = torch.ones(100, 3, 32, 32, 16)*0.1
            one_prob = torch.ones(100, 3, 32, 32, 16)*0.1

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.2, fraction)

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.int16)

            zero_prob = torch.ones(100, 3, 32, 32, 16)*0.5
            one_prob = torch.ones(100, 3, 32, 32, 16)*0

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.5, fraction)

    def testInt16MaskedIndividualRandomFlip(self):
        protected_bits = [0]*16

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.ones(100, 3, 32, 32, 16)
            one_prob = torch.ones(100, 3, 32, 32, 16)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            numpy.testing.assert_equal(numpy.logical_not(close), mask)

            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int16))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.zeros(100, 3, 32, 32, 16)
            one_prob = torch.zeros(100, 3, 32, 32, 16)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.numpy(), tensor.numpy())
            self.assertTrue(numpy.all(close))

    def testInt8Bits(self):
        tensor = torch.zeros((1), dtype=torch.int8)
        bits = common.torch.int_bits(tensor)
        assert not torch.any(bits)

        tensor = torch.ones((1), dtype=torch.int8)
        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '10000000')

        tensor = torch.ones((1), dtype=torch.int8)*(4+16+64+128)
        bits = common.torch.int_bits(tensor)
        # 0011001011010100
        self.checkBits(bits[0], '00101011')

    def testInt8HammingDistance(self):
        for i in range(10):
            torch_a = torch.CharTensor(1, 1).random_(0, torch.iinfo(torch.int8).max)
            torch_b = torch.CharTensor(1, 1).random_(0, torch.iinfo(torch.int8).max)

            numpy_a = torch_a.cpu().numpy()
            numpy_b = torch_b.cpu().numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
                for j in range(numpy_dist.shape[1]):
                    numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)

            numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())

    def testInt8HammingDistanceZero(self):
        for i in range(10):
            torch_a = torch.CharTensor(5, 5).fill_(0)
            torch_b = torch.CharTensor(5, 5).random_(0, torch.iinfo(torch.int8).max)

            numpy_a = torch_a.cpu().numpy()
            numpy_b = torch_b.cpu().numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
               for j in range(numpy_dist.shape[1]):
                   numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)
            numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())

    def testInt8And(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_and(numpy_a, numpy_b)

            torch_c = common.torch.int_and(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testInt8Or(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_or(numpy_a, numpy_b)

            torch_c = common.torch.int_or(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testInt8Xor(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.int8).max, size=(5, 5)).astype(numpy.int8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_xor(numpy_a, numpy_b)

            torch_c = common.torch.int_xor(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testInt8Flip(self):
        tensor = torch.tensor([0], dtype=torch.int8)
        mask = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0]]).bool()

        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '00000000')
        flipped_tensor = common.torch.int_flip(tensor, mask)
        flipped_bits = common.torch.int_bits(flipped_tensor)
        self.assertEqual(9, flipped_tensor[0])
        self.checkBits(flipped_bits[0], '10010000')

    def testInt8Set(self):
        for i in range(8):
            tensor = torch.tensor([0], dtype=torch.int8)
            bits = [0]*8
            bits[i] = 1
            set1 = torch.tensor([bits]).bool()
            set0 = torch.tensor([[0]*8]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '00000000')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            if i < 7:
                self.assertEqual(2**i, flipped_tensor[0].item())
            if i == 7:
                self.assertEqual(-2**7, flipped_tensor[0].item())
            self.assertTrue(flipped_bits[0][i])
            flipped_bits[0][i] = False
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 0)

        for i in range(8):
            tensor = torch.tensor([-1], dtype=torch.int8)
            bits = [0] * 8
            bits[i] = 1
            set0 = torch.tensor([bits]).bool()
            set1 = torch.tensor([[0] * 8]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '11111111')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            self.assertEqual(flipped_bits[0][i], False)
            flipped_bits[0][i] = True
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 8)

        for i in range(1000):
            tensor = torch.CharTensor(1).random_(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max)
            indices = numpy.random.choice(8, 2, replace=False)
            bits1 = [0] * 8
            bits1[indices[0]] = 1
            bits0 = [0] * 8
            bits0[indices[1]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(8):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], True)
                elif j == indices[1]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

        for i in range(100):
            tensor = torch.CharTensor(1).random_(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max)
            indices = numpy.random.choice(8, 1, replace=False)
            bits1 = [0] * 8
            bits1[indices[0]] = 1
            bits0 = [0] * 8
            bits0[indices[0]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(8):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

    def testInt8FlipHammingFixed(self):
        N = 3
        P = 8
        epsilon = 10

        tensor = torch.IntTensor(N).fill_(0).to(torch.int8)
        bits = common.torch.int_bits(tensor)
        mask = torch.BoolTensor(N * P).fill_(False)

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        self.assertEqual(flipped_tensor.dtype, tensor.dtype)
        self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

        indices = numpy.random.choice(N*P, size=epsilon, replace=False)
        #indices = [69, 22, 93, 95, 26, 29, 77, 44, 57, 14]
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        flipped_bits = common.torch.int_bits(flipped_tensor)
        # print(indices)
        # print(tensor, bits)
        # print(flipped_tensor, flipped_bits)
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt8FlipHamming(self):
        N = 10000
        P = 8
        epsilon = 500

        for i in range(10):
            tensor = torch.CharTensor(N).random_(0, torch.iinfo(torch.int8).max)
            mask = torch.BoolTensor(N * P).fill_(False)

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
            self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

            indices = numpy.random.choice(N * P, size=epsilon, replace=False)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True
            self.assertEqual(epsilon, torch.sum(mask).item())

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))

            self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testInt8RandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertFalse(numpy.all(close))
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.45, fraction)

    def testInt8RandomFlipNaNInf(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            self.assertFalse(numpy.any(numpy.isnan(flipped_tensor.cpu().numpy())))
            self.assertFalse(numpy.any(numpy.isinf(flipped_tensor.cpu().numpy())))

    def testInt8RandomFlipProtected(self):
        protected_bits = [1]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))
            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)
            numpy.testing.assert_almost_equal(tensor.cpu().numpy(), flipped_tensor.cpu().numpy())

    def testInt8MaskedRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))
            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))

            flipped_tensor = common.torch.int_masked_random_flip(tensor, mask, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertFalse(numpy.all(close))

            numpy.testing.assert_almost_equal((1 - mask.cpu().numpy())*tensor.cpu().numpy(), (1 - mask.cpu().numpy())*flipped_tensor.cpu().numpy())

    def testInt8IndividualRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.int8)

            zero_prob = torch.ones(100, 3, 32, 32, 8)*0.1
            one_prob = torch.ones(100, 3, 32, 32, 8)*0.1

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.5, fraction)

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.int8)

            zero_prob = torch.ones(100, 3, 32, 32, 8)*0.5
            one_prob = torch.ones(100, 3, 32, 32, 8)*0

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.2, fraction)

    def testInt8MaskedIndividualRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.ones(100, 3, 32, 32, 8)
            one_prob = torch.ones(100, 3, 32, 32, 8)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            numpy.testing.assert_equal(numpy.logical_not(close), mask.cpu().numpy())

            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.int8))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.zeros(100, 3, 32, 32, 8)
            one_prob = torch.zeros(100, 3, 32, 32, 8)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertTrue(numpy.all(close))

    def testUInt8Bits(self):
        tensor = torch.zeros((1), dtype=torch.uint8)
        bits = common.torch.int_bits(tensor)
        assert not torch.any(bits)

        tensor = torch.ones((1), dtype=torch.uint8)
        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '10000000')

        tensor = torch.ones((1), dtype=torch.uint8)*(4+16+64+128)
        bits = common.torch.int_bits(tensor)
        # 0011001011010100
        self.checkBits(bits[0], '00101011')

    def testUInt8MSBProjection(self):
        def test(original, perturbed):
            projected = common.torch.int_msb_projection(original, perturbed)

            original_bits = common.torch.int_bits(original)
            perturbed_bits = common.torch.int_bits(perturbed)
            projected_bits = common.torch.int_bits(projected)

            perturbed_dist = common.torch.int_hamming_distance(original, perturbed)
            projected_dist = common.torch.int_hamming_distance(original, projected)

            #print('---')
            #print(original.item(), '\t', ''.join(list(map(str, list(original_bits[0].int().numpy())))))
            #print(perturbed.item(), '\t', ''.join(list(map(str, list(perturbed_bits[0].int().numpy())))), perturbed_dist.item())
            #print(projected.item(), '\t', ''.join(list(map(str, list(projected_bits[0].int().numpy())))), projected_dist.item())

            self.assertGreaterEqual(1, projected_dist.item())
            if perturbed_dist.item() == 0:
                self.assertEqual(0, projected_dist.item())

            counter = 0
            for i in range(8):
                if original_bits[0, i] != projected_bits[0, i]:
                    self.assertEqual(0, counter)
                    counter += 1

            return projected

        original = torch.IntTensor([15]).to(torch.uint8)
        perturbed = torch.IntTensor([7]).to(torch.uint8)
        projected = test(original, perturbed)

        original = torch.IntTensor([15]).to(torch.uint8)
        perturbed = torch.IntTensor([6]).to(torch.uint8)
        projected = test(original, perturbed)

    def testUInt8HammingDistance(self):
        for i in range(10):
            torch_a = torch.ByteTensor(1, 1).random_(0, torch.iinfo(torch.uint8).max)
            torch_b = torch.ByteTensor(1, 1).random_(0, torch.iinfo(torch.uint8).max)

            numpy_a = torch_a.cpu().numpy()
            numpy_b = torch_b.cpu().numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
                for j in range(numpy_dist.shape[1]):
                    numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)

            numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())

    def testUInt8HammingDistanceZero(self):
        for i in range(10):
            torch_a = torch.ByteTensor(5, 5).fill_(0)
            torch_b = torch.ByteTensor(5, 5).random_(0, torch.iinfo(torch.uint8).max)

            numpy_a = torch_a.cpu().numpy()
            numpy_b = torch_b.cpu().numpy()

            numpy_xor = numpy.bitwise_xor(numpy_a, numpy_b)
            numpy_dist = numpy.zeros((numpy_a.shape[0], numpy_a.shape[1]))
            for i in range(numpy_dist.shape[0]):
               for j in range(numpy_dist.shape[1]):
                   numpy_dist[i, j] = numpy.binary_repr(numpy_xor[i, j]).count('1')

            torch_dist = common.torch.int_hamming_distance(torch_a, torch_b)
            numpy.testing.assert_equal(numpy_dist, torch_dist.cpu().numpy())

    def testUInt8And(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_and(numpy_a, numpy_b)

            torch_c = common.torch.int_and(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testUInt8Or(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_or(numpy_a, numpy_b)

            torch_c = common.torch.int_or(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testUInt8Xor(self):
        for i in range(10):
            numpy_a = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)
            numpy_b = numpy.random.randint(0, torch.iinfo(torch.uint8).max, size=(5, 5)).astype(numpy.uint8)

            torch_a = torch.from_numpy(numpy_a)
            torch_b = torch.from_numpy(numpy_b)

            numpy_c = numpy.bitwise_xor(numpy_a, numpy_b)

            torch_c = common.torch.int_xor(torch_a, torch_b)

            numpy.testing.assert_almost_equal(numpy_c, torch_c.cpu().numpy())

    def testUInt8Flip(self):
        tensor = torch.tensor([0], dtype=torch.uint8)
        mask = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0]]).bool()

        bits = common.torch.int_bits(tensor)
        self.checkBits(bits[0], '00000000')
        flipped_tensor = common.torch.int_flip(tensor, mask)
        flipped_bits = common.torch.int_bits(flipped_tensor)
        self.assertEqual(9, flipped_tensor[0])
        self.checkBits(flipped_bits[0], '10010000')

    def testUInt8Set(self):
        for i in range(8):
            tensor = torch.tensor([0], dtype=torch.uint8)
            bits = [0]*8
            bits[i] = 1
            set1 = torch.tensor([bits]).bool()
            set0 = torch.tensor([[0]*8]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '00000000')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            self.assertEqual(2**i, flipped_tensor[0].item())
            self.assertTrue(flipped_bits[0][i])
            flipped_bits[0][i] = False
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 0)

        for i in range(8):
            tensor = torch.tensor([-1], dtype=torch.uint8)
            bits = [0] * 8
            bits[i] = 1
            set0 = torch.tensor([bits]).bool()
            set1 = torch.tensor([[0] * 8]).bool()

            bits = common.torch.int_bits(tensor)
            self.checkBits(bits[0], '11111111')
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            self.assertEqual(flipped_bits[0][i], False)
            flipped_bits[0][i] = True
            self.assertEqual(torch.sum(flipped_bits[0].int()).item(), 8)

        for i in range(1000):
            tensor = torch.ByteTensor(1).random_(0, torch.iinfo(torch.uint8).max)
            indices = numpy.random.choice(8, 2, replace=False)
            bits1 = [0] * 8
            bits1[indices[0]] = 1
            bits0 = [0] * 8
            bits0[indices[1]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(8):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], True)
                elif j == indices[1]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

        for i in range(100):
            tensor = torch.ByteTensor(1).random_(0, torch.iinfo(torch.uint8).max)
            indices = numpy.random.choice(8, 1, replace=False)
            bits1 = [0] * 8
            bits1[indices[0]] = 1
            bits0 = [0] * 8
            bits0[indices[0]] = 1
            set1 = torch.tensor([bits1]).bool()
            set0 = torch.tensor([bits0]).bool()

            bits = common.torch.int_bits(tensor)
            flipped_tensor = common.torch.int_set(tensor, set1, set0)
            flipped_bits = common.torch.int_bits(flipped_tensor)
            for j in range(8):
                if j == indices[0]:
                    self.assertEqual(flipped_bits[0][j], False)
                else:
                    self.assertEqual(bits[0][j], flipped_bits[0][j])

    def testUInt8FlipHammingFixed(self):
        N = 3
        P = 8
        epsilon = 10

        tensor = torch.IntTensor(N).fill_(0).to(torch.uint8)
        bits = common.torch.int_bits(tensor)
        mask = torch.BoolTensor(N * P).fill_(False)

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        self.assertEqual(flipped_tensor.dtype, tensor.dtype)
        self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

        indices = numpy.random.choice(N*P, size=epsilon, replace=False)
        #indices = [69, 22, 93, 95, 26, 29, 77, 44, 57, 14]
        numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
        mask[indices] = True
        self.assertEqual(epsilon, torch.sum(mask).item())

        flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
        flipped_bits = common.torch.int_bits(flipped_tensor)
        # print(indices)
        # print(tensor, bits)
        # print(flipped_tensor, flipped_bits)
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))
        # print(common.torch.int_hamming_distance(tensor, flipped_tensor))

        self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testUInt8FlipHamming(self):
        N = 10000
        P = 8
        epsilon = 500

        for i in range(10):
            tensor = torch.ByteTensor(N).random_(0, torch.iinfo(torch.uint8).max)
            mask = torch.BoolTensor(N * P).fill_(False)

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))
            self.assertEqual(0, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

            indices = numpy.random.choice(N * P, size=epsilon, replace=False)
            numpy.testing.assert_array_equal(numpy.sort(indices), numpy.unique(indices))
            mask[indices] = True
            self.assertEqual(epsilon, torch.sum(mask).item())

            flipped_tensor = common.torch.int_flip(tensor, mask.view(N, P))

            self.assertEqual(epsilon, torch.sum(common.torch.int_hamming_distance(tensor, flipped_tensor)).item())

    def testUInt8RandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertFalse(numpy.all(close))
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.45, fraction)

    def testUInt8RandomFlipNaNInf(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))

            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)

            self.assertFalse(numpy.any(numpy.isnan(flipped_tensor.cpu().numpy())))
            self.assertFalse(numpy.any(numpy.isinf(flipped_tensor.cpu().numpy())))

    def testUInt8RandomFlipProtected(self):
        protected_bits = [1]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))
            flipped_tensor = common.torch.int_random_flip(tensor, 0.1, 0.1, protected_bits)
            numpy.testing.assert_almost_equal(tensor.cpu().numpy(), flipped_tensor.cpu().numpy())

    def testUInt8MaskedRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))
            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))

            flipped_tensor = common.torch.int_masked_random_flip(tensor, mask, 0.1, 0.1, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertFalse(numpy.all(close))

            numpy.testing.assert_almost_equal((1 - mask.cpu().numpy())*tensor.cpu().numpy(), (1 - mask.cpu().numpy())*flipped_tensor.cpu().numpy())

    def testUInt8IndividualRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.uint8)

            zero_prob = torch.ones(100, 3, 32, 32, 8)*0.1
            one_prob = torch.ones(100, 3, 32, 32, 8)*0.1

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.5, fraction)

        for i in range(10):
            tensor = torch.randn(100, 3, 32, 32).to(torch.uint8)

            zero_prob = torch.ones(100, 3, 32, 32, 8)*0.5
            one_prob = torch.ones(100, 3, 32, 32, 8)*0

            flipped_tensor = common.torch.int_individual_random_flip(tensor, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            fraction = numpy.sum(close) / (float(numpy.prod(tensor.shape)))
            self.assertGreaterEqual(0.2, fraction)

    def testUInt8MaskedIndividualRandomFlip(self):
        protected_bits = [0]*8

        for i in range(10):
            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.ones(100, 3, 32, 32, 8)
            one_prob = torch.ones(100, 3, 32, 32, 8)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            numpy.testing.assert_equal(numpy.logical_not(close), mask.cpu().numpy())

            tensor = torch.from_numpy(numpy.random.randint(0, 100, (100, 3, 32, 32)).astype(numpy.uint8))

            mask = torch.from_numpy(numpy.random.randint(0, 2, (100, 3, 32, 32)).astype(bool))
            zero_prob = torch.zeros(100, 3, 32, 32, 8)
            one_prob = torch.zeros(100, 3, 32, 32, 8)

            flipped_tensor = common.torch.int_masked_individual_random_flip(tensor, mask, zero_prob, one_prob, protected_bits)

            close = numpy.isclose(flipped_tensor.cpu().numpy(), tensor.cpu().numpy())
            self.assertTrue(numpy.all(close))

if __name__ == '__main__':
    unittest.main()
