import unittest
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.test
import common.train
import common.datasets
import common.torch
import common.state
import common.eval
import common.quantization
import common.progress
import attacks
import attacks.weights
import torch
import torch.utils.data


class TestAttacksWeightsMNISTMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 100
        cls.cuda = True
        cls.setDatasets()

        if os.path.exists(cls.getModelFile()):
            state = common.state.State.load(cls.getModelFile())
            cls.model = state.model

            if cls.cuda:
                cls.model = cls.model.cuda()
        else:
            cls.model = cls.getModel()
            if cls.cuda:
                cls.model = cls.model.cuda()

            optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1, momentum=0.9)
            scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(cls.trainloader))
            writer = common.summary.SummaryWriter()
            augmentation = None

            trainer = common.train.NormalTraining(cls.model, cls.trainloader, cls.testloader, optimizer, scheduler, augmentation=augmentation, writer=writer,
                                                  cuda=cls.cuda)
            for e in range(5):
                trainer.step(e)

            common.state.State.checkpoint(cls.getModelFile(), cls.model, optimizer, scheduler, e)

            cls.model.eval()
            probabilities = common.test.test(cls.model, cls.testloader, cuda=cls.cuda)
            eval = common.eval.CleanEvaluation(probabilities, cls.testloader.dataset.labels, validation=0)
            assert 0.075 > eval.test_error(), '0.05 !> %g' % eval.test_error()
            assert numpy.mean(numpy.max(probabilities, axis=1)) > 0.9

        cls.model.eval()

    @classmethod
    def setDatasets(cls):
        cls.trainset = common.datasets.MNISTTrainSet(indices=range(10000))
        cls.testset = common.datasets.MNISTTestSet(indices=range(1000))
        cls.adversarialset = common.datasets.MNISTTestSet(indices=range(100))
        cls.trainloader = torch.utils.data.DataLoader(cls.trainset, batch_size=cls.batch_size, shuffle=True, num_workers=0)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=cls.batch_size, shuffle=False, num_workers=0)
        cls.adversarialloader = torch.utils.data.DataLoader(cls.adversarialset, batch_size=cls.batch_size, shuffle=False, num_workers=0)

    @classmethod
    def getModelFile(cls):
        return 'mnist_mlp.pth.tar'

    @classmethod
    def getModel(cls):
        return models.MLP(10, [1, 28, 28], units=[100, 100, 100], action=torch.nn.Sigmoid)

    def _testAttackPerformance(self, attack, attempts=5, objective=attacks.weights.objectives.UntargetedF0Objective()):
        error_rate = 0
        for t in range(attempts):
            perturbed_model = attack.run(self.model, self.adversarialloader, objective)
            perturbed_model = perturbed_model.cuda()

            probabilities = common.test.test(perturbed_model, self.adversarialloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, self.adversarialset.labels)
            error_rate += eval.test_error()

        error_rate /= attempts
        return error_rate

    def _testQuantizedAttackPerformance(self, attack, attempts=5, objective=attacks.weights.objectives.UntargetedF0Objective()):
        assert attack.quantization is not None

        error_rate = 0
        quantized_model, contexts = common.quantization.quantize(attack.quantization, self.model)

        for t in range(attempts):
            perturbed_model = attack.run(quantized_model, self.adversarialloader, objective)
            perturbed_model = perturbed_model.cuda()

            probabilities = common.test.test(perturbed_model, self.adversarialloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, self.adversarialset.labels)
            error_rate += eval.test_error()

        error_rate /= attempts
        return error_rate

    def testHessianAttack(self):
        epsilon = 0.001
        attack = attacks.weights.HessianAttack()
        attack.progress = common.progress.ProgressBar()
        attack.k = 0
        attack.initialization = None#attacks.weights.initializations.LInfUniformNormInitialization(epsilon)
        attack.projection = attacks.weights.projections.LInfProjection(relative_epsilon=epsilon)
        attack.norm = attacks.weights.norms.LInfNorm()

        error_rate = self._testAttackPerformance(attack)
        self.assertGreaterEqual(error_rate, 0.8)

    def testRandomAttackLInf(self):
        error_rates = []
        for epsilon in [0.05, 0.1, 0.25]:
            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.weights.projections.LInfProjection(epsilon)
            attack.norm = attacks.weights.norms.LInfNorm()
            error_rates.append(self._testAttackPerformance(attack))

        for error_rate in error_rates:
            self.assertGreaterEqual(error_rate, 0.01)
        for i in range(1, len(error_rates)):
            self.assertGreaterEqual(error_rates[i], error_rates[i - 1])

    def testQuantizedRandomAttackBit(self):
        for bits in [8, 16, 32]:
            error_rates = []
            max_val = 2
            quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=max_val, precision=bits)

            for probability in [0.0001, 0.001]:
                attack = attacks.weights.RandomAttack()
                attack.progress = common.progress.ProgressBar()
                attack.quantization = quantization
                attack.epochs = 10
                attack.initialization = attacks.weights.initializations.BitRandomInitialization(probability)
                attack.projection = attacks.weights.projections.BoxProjection(-max_val, max_val)
                attack.norm = attacks.weights.norms.HammingNorm()
                error_rates.append(self._testQuantizedAttackPerformance(attack))

            for error_rate in error_rates:
                self.assertGreaterEqual(error_rate, 0.01)
            for i in range(1, len(error_rates)):
                self.assertGreaterEqual(error_rates[i], error_rates[i - 1])

    def testAdaptiveQuantizedRandomAttackBit(self):
        for bits in [8, 16, 32]:
            max_val = 2

            probability = 0.001
            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=max_val, precision=bits)
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.BitRandomInitialization(probability)
            attack.projection = attacks.weights.projections.BoxProjection(-max_val, max_val)
            attack.norm = attacks.weights.norms.HammingNorm()
            fixed_error_rate = self._testQuantizedAttackPerformance(attack)

            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision=bits)
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.BitRandomInitialization(probability)
            attack.projection = attacks.weights.projections.BoxProjection(-max_val, max_val)
            attack.norm = attacks.weights.norms.HammingNorm()
            adaptive_error_rate = self._testQuantizedAttackPerformance(attack)

            self.assertGreaterEqual(fixed_error_rate, adaptive_error_rate)

    def testQuantizedRandomAttackBitPattern(self):
        for bits in [8, 16, 32]:
            max_val = 2
            quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=max_val, precision=bits)

            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.quantization = quantization
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.BitPatternInitialization(0.01, 1)
            attack.projection = attacks.weights.projections.BoxProjection(-max_val, max_val)
            attack.norm = attacks.weights.norms.HammingNorm()
            error_rate = self._testQuantizedAttackPerformance(attack)
            print(error_rate)

    def testAdaptiveQuantizedRandomAttackBitPattern(self):
        for bits in [8, 16, 32]:
            quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision=bits)

            attack = attacks.weights.RandomAttack()
            attack.progress = common.progress.ProgressBar()
            attack.quantization = quantization
            attack.epochs = 10
            attack.initialization = attacks.weights.initializations.BitPatternInitialization(0.01, 1)
            attack.projection = None
            attack.norm = attacks.weights.norms.HammingNorm()
            error_rate = self._testQuantizedAttackPerformance(attack)
            print(error_rate)


class TestAttacksWeightsMNISTResNet(TestAttacksWeightsMNISTMLP):
    @classmethod
    def getModelFile(cls):
        return 'mnist_resnet.pth.tar'

    @classmethod
    def getModel(cls):
        return models.ResNet(10, [1, 28, 28], blocks=[1, 1, 1])


if __name__ == '__main__':
    unittest.main()
