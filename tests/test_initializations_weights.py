import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.numpy
import common.state
import common.eval
import common.quantization
import common.utils
import attacks
import torch
import torch.utils.data
import math 


class TestNet(torch.nn.Module):
    def __init__(self, D=100, K=10, L=10):
        super(TestNet, self).__init__()
        self.L = L
        Ds = []
        for l in range(self.L):
            Ds.append(D + l)
        for l in range(self.L):
            linear = torch.nn.Linear(Ds[l], Ds[l], bias=False)
            torch.nn.init.uniform_(linear.weight, -1, 1)
            setattr(self, 'linear%d' % l, linear)
        self.logits = torch.nn.Linear(Ds[-1], K, bias=False)
        torch.nn.init.uniform_(self.logits.weight, -1, 1)
        # !
        self.linear0.weight.requires_grad = False

    def forward(self, inputs):
        for l in range(self.L):
            linear = getattr(self, 'linear%d' % l)
            inputs = linear(inputs)
        return self.logits(inputs)


class TestInitializationsWeights(unittest.TestCase):
    def testLInfUniformNormInitialization(self):
        model = TestNet()

        N = 5
        epsilon = 0.01
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            layers = list(range(len(list(perturbed_model.parameters()))))
            norm = attacks.weights.norms.LInfNorm()
            dist = norm(model, perturbed_model, layers)
            self.assertAlmostEqual(0, dist)
            initialization = attacks.weights.initializations.LInfUniformNormInitialization(epsilon)
            initialization(model, perturbed_model, layers)
            dist = norm(model, perturbed_model, layers)
            self.assertGreaterEqual(epsilon, dist)
            self.assertGreater(dist, 0)

    def testL0UniformNormInitialization(self):
        model = TestNet()

        N = 5
        epsilon = 100
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            layers = list(range(len(list(perturbed_model.parameters()))))
            norm = attacks.weights.norms.L0Norm()
            dist = norm(model, perturbed_model, layers)
            self.assertEqual(0, dist)
            initialization = attacks.weights.initializations.L0UniformNormInitialization(epsilon)
            initialization(model, perturbed_model, layers)
            dist = norm(model, perturbed_model, layers)
            self.assertGreaterEqual(epsilon, dist)
            self.assertGreater(dist, 0)

    def testL0UniformNormInitializationDistribution(self):
        model = TestNet()

        N = 100
        epsilon = 100
        histogram = [0]*10
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            layers = list(range(len(list(perturbed_model.parameters()))))
            norm = attacks.weights.norms.L0Norm()
            dist = norm(model, perturbed_model, layers)
            self.assertEqual(0, dist)
            initialization = attacks.weights.initializations.L0UniformNormInitialization(epsilon)
            initialization(model, perturbed_model, layers)
            dist = norm(model, perturbed_model, layers)
            bin = int(dist - 1)//10
            histogram[bin] += 1

    def testL0UniformNormInitializationProbability(self):
        model = TestNet()
        n, _, _, _ = common.torch.parameter_sizes(model)

        N = 5
        probability = 0.1
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            norm = attacks.weights.norms.L0Norm()
            initialization = attacks.weights.initializations.L0UniformNormInitialization(epsilon=None, probability=probability)
            layers = list(range(len(list(perturbed_model.parameters()))))
            initialization(model, perturbed_model, layers)
            dist = norm(model, perturbed_model, layers)
            epsilon = int(n*probability) 
            self.assertEqual(initialization.epsilon, epsilon)
            self.assertGreaterEqual(epsilon, dist)
            self.assertGreater(dist, 0)

    def testL0RandomInitialization(self):
        model = TestNet()
        n, _, _, _ = common.torch.parameter_sizes(model)

        N = 100
        probability = 0.1
        dist = 0.
        for i in range(N):
            perturbed_model = common.torch.clone(model)
            norm = attacks.weights.norms.L0Norm()
            initialization = attacks.weights.initializations.L0RandomInitialization(probability=probability)
            layers = list(range(len(list(perturbed_model.parameters()))))
            initialization(model, perturbed_model, layers)
            dist += norm(model, perturbed_model, layers)
        dist /= N
        epsilon = int(n*probability)
        self.assertGreaterEqual(epsilon + 100, dist)
        self.assertGreaterEqual(dist, epsilon - 100)

    def testL0RandomInitializationRandomness(self):
        model = TestNet()

        N = 5
        randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(N)))))

        dist_0 = numpy.zeros((10, N))
        dist_inf = numpy.zeros((10, N))
        for i in range(10):
            randomloader = torch.utils.data.DataLoader(randomset, shuffle=False, batch_size=1)
            initialization = attacks.weights.initializations.L0RandomInitialization(0.01, randomloader)
            for j in range(N):
                perturbed_model = common.torch.clone(model)
                norm_0 = attacks.weights.norms.L0Norm()
                norm_inf = attacks.weights.norms.LInfNorm()
                layers = list(range(len(list(perturbed_model.parameters()))))
                # includes asserts!
                initialization(model, perturbed_model, layers)
                dist_0[i, j] = norm_0(model, perturbed_model, layers)
                dist_inf[i, j] = norm_inf(model, perturbed_model, layers)

        for j in range(N):
            self.assertEqual(0, numpy.sum(dist_0[:, j] - dist_0[0, j]))
            self.assertAlmostEqual(0, numpy.sum(dist_inf[:, j] - dist_inf[0, j]))

    def hamming_distance(self, quantization, quantization_contexts, model, perturbed_model):
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        dist = 0
        for i in range(len(parameters)):
            quantized_parameter, _ = quantization.quantize(parameters[i], quantization_contexts[i])
            quantized_perturbed_parameter, _ = quantization.quantize(perturbed_parameters[i], quantization_contexts[i])
            dist += torch.sum(common.torch.int_hamming_distance(quantized_parameter, quantized_perturbed_parameter)).item()

        return dist

    def testRandomBitInitialization(self):
        model = TestNet()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)

        N = 20
        dist = 0
        for i in range(N):
            perturbed_model = common.torch.clone(model)

            self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
            initialization = attacks.weights.initializations.BitRandomInitialization(0.01)

            layers = list(range(len(list(perturbed_model.parameters()))))
            # includes asserts!
            initialization(model, perturbed_model, layers, quantization, quantization_contexts)
            dist += self.hamming_distance(quantization, quantization_contexts, model, perturbed_model)

        dist /= N
        n, _, _, _ = common.torch.parameter_sizes(model)
        n *= 16
        self.assertGreaterEqual(0.015, dist/n)
        self.assertGreaterEqual(dist/n, 0.005)

    def testRandomBitInitializationRandomness(self):
        model = TestNet()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)

        N = 2
        randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(N)))))

        dist = numpy.zeros((10, N))
        for i in range(10):
            randomloader = torch.utils.data.DataLoader(randomset, shuffle=False, batch_size=1)
            initialization = attacks.weights.initializations.BitRandomInitialization(0.01, randomloader)
            for j in range(N):
                perturbed_model = common.torch.clone(model)
                self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
                layers = list(range(len(list(perturbed_model.parameters()))))
                # includes asserts!
                initialization(model, perturbed_model, layers, quantization, quantization_contexts)
                dist[i, j] = self.hamming_distance(quantization, quantization_contexts, model, perturbed_model)

        for j in range(N):
            self.assertEqual(0, numpy.sum(dist[:, j] - dist[0, j]))

    def testRandomBitMSBInitializationRandomness(self):
        model = TestNet()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)

        N = 2
        randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(N)))))

        dist = numpy.zeros((10, N))
        for i in range(10):
            randomloader = torch.utils.data.DataLoader(randomset, shuffle=False, batch_size=1)
            initialization = attacks.weights.initializations.BitRandomMSBInitialization(0.01, randomloader)
            for j in range(N):
                perturbed_model = common.torch.clone(model)
                self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
                layers = list(range(len(list(perturbed_model.parameters()))))
                # includes asserts!
                initialization(model, perturbed_model, layers, quantization, quantization_contexts)
                dist[i, j] = self.hamming_distance(quantization, quantization_contexts, model, perturbed_model)

        for j in range(N):
            self.assertEqual(0, numpy.sum(dist[:, j] - dist[0, j]))

    def testRandomBitInitializationProbability(self):
        model = TestNet(D=200, L=25)#.cuda()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)
        # attack expects a quantized model as input, then quantizes it again to get the contexts
        # perturbed model is then a copy of the quantized model

        dist = 0
        N = 100
        n, _, _, _ = common.torch.parameter_sizes(model)
        norm = attacks.weights.norms.HammingNorm()
        norm2 = attacks.weights.norms.L0Norm()
        probability = 0.05

        for i in range(N):
            perturbed_model = common.torch.clone(model)

            self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
            initialization = attacks.weights.initializations.BitRandomInitialization(probability)
            initialization.debug = True

            layers = list(range(len(list(perturbed_model.parameters()))))
            # includes asserts!
            initialization(model, perturbed_model, layers, quantization, quantization_contexts)
            dist_i = norm(model, perturbed_model, layers, quantization, quantization_contexts)
            dist2_i = norm2(model, perturbed_model, layers)
            dist += dist_i

        dist /= N
        dist /= n*quantization.type_precision
        self.assertAlmostEqual(dist, probability, 3)

    def testRandomBitInitializationProbability6Bit(self):
        model = TestNet(D=200, L=25)#.cuda()
        quantization = common.quantization.AlternativeUnsignedFixedPointQuantization(precision=6, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)
        # attack expects a quantized model as input, then quantizes it again to get the contexts
        # perturbed model is then a copy of the quantized model

        dist = 0
        N = 100
        n, _, _, _ = common.torch.parameter_sizes(model)
        norm = attacks.weights.norms.HammingNorm()
        norm2 = attacks.weights.norms.L0Norm()
        probability = 0.05

        for i in range(N):
            perturbed_model = common.torch.clone(model)

            self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
            initialization = attacks.weights.initializations.BitRandomInitialization(probability)
            initialization.debug = True

            layers = list(range(len(list(perturbed_model.parameters()))))
            # includes asserts!
            initialization(model, perturbed_model, layers, quantization, quantization_contexts)
            dist_i = norm(model, perturbed_model, layers, quantization, quantization_contexts)
            dist2_i = norm2(model, perturbed_model, layers)
            dist += dist_i

        dist /= N
        dist /= n*quantization.repr_precision
        self.assertAlmostEqual(dist, probability, 2)

    def testRandomBitMSBInitializationProbability(self):
        model = TestNet(D=200, L=25)#.cuda()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)
        # attack expects a quantized model as input, then quantizes it again to get the contexts
        # perturbed model is then a copy of the quantized model

        dist = 0
        N = 100
        n, _, _, _ = common.torch.parameter_sizes(model)
        norm = attacks.weights.norms.HammingNorm()
        norm2 = attacks.weights.norms.L0Norm()
        norm3 = attacks.weights.norms.LInfNorm()
        probability = 0.05

        for i in range(N):
            perturbed_model = common.torch.clone(model)

            self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
            initialization = attacks.weights.initializations.BitRandomMSBInitialization(probability)
            initialization.debug = True

            layers = list(range(len(list(perturbed_model.parameters()))))
            # includes asserts!
            initialization(model, perturbed_model, layers, quantization, quantization_contexts)
            dist_i = norm(model, perturbed_model, layers, quantization, quantization_contexts)
            dist2_i = norm2(model, perturbed_model, layers)
            dist3_i = norm3(model, perturbed_model, layers)
            self.assertAlmostEqual(dist3_i, 1, places=2)
            dist += dist_i

        dist /= N
        dist /= n*quantization.type_precision
        self.assertAlmostEqual(dist, probability/16, 3)

    def testRandomBitLSBInitializationProbability(self):
        model = TestNet(D=200, L=25)#.cuda()
        quantization = common.quantization.AlternativeFixedPointQuantization(precision=16, max_abs_range=1)
        model, quantization_contexts = common.quantization.quantize(quantization, model)
        # attack expects a quantized model as input, then quantizes it again to get the contexts
        # perturbed model is then a copy of the quantized model

        dist = 0
        N = 100
        lsb = 14
        n, _, _, _ = common.torch.parameter_sizes(model)
        norm = attacks.weights.norms.HammingNorm()
        norm2 = attacks.weights.norms.L0Norm()
        norm3 = attacks.weights.norms.LInfNorm()
        probability = 0.05

        for i in range(N):
            perturbed_model = common.torch.clone(model)

            self.assertEqual(0, self.hamming_distance(quantization, quantization_contexts, model, perturbed_model))
            initialization = attacks.weights.initializations.BitRandomLSBInitialization(probability, lsb=lsb)
            initialization.debug = True

            layers = list(range(len(list(perturbed_model.parameters()))))
            # includes asserts!
            initialization(model, perturbed_model, layers, quantization, quantization_contexts)
            dist_i = norm(model, perturbed_model, layers, quantization, quantization_contexts)
            dist2_i = norm2(model, perturbed_model, layers)
            dist3_i = norm3(model, perturbed_model, layers)
            self.assertGreaterEqual(0.5, dist3_i)
            dist += dist_i

        dist /= N
        dist /= n*quantization.type_precision
        self.assertAlmostEqual(dist, lsb*probability/16, 3)


if __name__ == '__main__':
    unittest.main()
