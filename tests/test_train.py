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
import common.state
import common.imgaug
import common.eval
import common.quantization
import attacks
import attacks.weights
import torch
import torch.utils.data
from imgaug import augmenters as iaa


class TestTrainMNIST(unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.trainset = torch.utils.data.DataLoader(common.datasets.MNISTTrainSet(indices=range(10000)), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.testset = torch.utils.data.DataLoader(common.datasets.MNISTTestSet(indices=range(5000)), batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.cuda = True
        self.model = models.LeNet(10, [1, 28, 28], channels=32, normalization='bn', linear=256)

        if self.cuda:
            self.model = self.model.cuda()

    def testNormalTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testQuantizedNormalTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        quantization = common.quantization.AlternativeFixedPointQuantization(max_abs_range=2, precision=8)
        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.quantization = quantization

        epochs = 10
        for e in range(epochs):
            probabilities, forward_model, contexts = trainer.step(e)

        probabilities = common.test.test(forward_model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

        quantized_forward_model, _ = common.quantization.quantize(quantization, forward_model)
        self.assertAlmostEqual(0, common.quantization.error(forward_model, quantized_forward_model))
        self.assertGreaterEqual(common.quantization.error(forward_model, self.model), 0.01)

    def testNormalTrainingAugmentation(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.025)),
            iaa.Add((-0.075, 0.075)),
            common.imgaug.Clip(0, 1)
        ])

        trainer = common.train.NormalTraining(self.model, self.trainset, self.testset, optimizer, scheduler, augmentation=augmentation, writer=writer, cuda=self.cuda)

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        self.assertGreaterEqual(0.05, eval.test_error())

    def testAverageWeightsTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        objective = attacks.weights.UntargetedF0Objective()
        epsilon = 0.3
        clipping = 1
        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(epsilon)
        attack.projection = attacks.weights.projections.SequentialProjections([
            attacks.weights.projections.BoxProjection(-clipping, clipping),
            attacks.weights.projections.LayerWiseL2Projection(epsilon)
        ])
        attack.norm = attacks.weights.norms.L2Norm()

        trainer = common.train.AverageWeightsTraining(self.model, self.trainset, self.testset, optimizer, scheduler, attack, objective, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.projection = attacks.weights.projections.BoxProjection(-clipping, clipping)

        def simple_curriculum(attack, loss, perturbed_loss, epoch):
            if perturbed_loss < 2.15:
                population = 1
            else:
                population = 0

            return population, {
                'population': population,
                'epochs': attack.epochs,
            }

        trainer.curriculum = simple_curriculum

        epochs = 10
        for e in range(epochs):
            trainer.step(e)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        print(probabilities)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        #self.assertGreaterEqual(0.05, eval.test_error())
        print(eval.test_error())

        perturbed_probabilities = None
        for b, (inputs, targets) in enumerate(self.testset):
            batchset = [(inputs, targets)]
            perturbed_model = attack.run(self.model, batchset, objective)
            if self.cuda:
                perturbed_model = perturbed_model.cuda()
            perturbed_probabilities = common.numpy.concatenate(perturbed_probabilities, common.test.test(perturbed_model, batchset, cuda=self.cuda))

        eval = common.eval.CleanEvaluation(perturbed_probabilities, self.testset.dataset.labels, validation=0)
        print(eval.test_error())

    def testAdversarialWeightsTraining(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainset))
        writer = common.summary.SummaryDictWriter()
        augmentation = None

        objective = attacks.weights.UntargetedF0Objective()
        epsilon = 0.05
        clipping = 1
        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.initialization = attacks.weights.initializations.LayerWiseL2UniformNormInitialization(epsilon)
        attack.projection = attacks.weights.projections.SequentialProjections([
            attacks.weights.projections.BoxProjection(-clipping, clipping),
            attacks.weights.projections.LayerWiseL2Projection(epsilon)
        ])
        attack.norm = attacks.weights.norms.L2Norm()

        trainer = common.train.AdversarialWeightsTraining(self.model, self.trainset, self.testset, optimizer, scheduler,
                                                          attack, objective, augmentation=augmentation, writer=writer, cuda=self.cuda)
        trainer.projection = attacks.weights.projections.BoxProjection(-clipping, clipping)

        def simple_curriculum(attack, loss, perturbed_loss, epoch):
            if perturbed_loss < 0.5:
                attack.epochs = 20
            elif perturbed_loss < 1:
                attack.epochs = 15
            elif perturbed_loss < 1.5:
                attack.epochs = 10
            elif perturbed_loss < 1.75:
                attack.epochs = 7
            elif perturbed_loss < 2:
                attack.epochs = 5
            elif perturbed_loss < 2.15:
                attack.epochs = 3
            else:
                attack.epochs = 1
            population = 1

            return population, {
                'population': population,
                'epochs': attack.epochs,
            }
        trainer.curriculum = simple_curriculum

        epochs = 10
        for e in range(epochs):
            trainer.step(e)
            common.state.State.checkpoint('test.pth.tar', self.model)

        probabilities = common.test.test(self.model, self.testset, cuda=self.cuda)
        eval = common.eval.CleanEvaluation(probabilities, self.testset.dataset.labels, validation=0)
        #self.assertGreaterEqual(0.05, eval.test_error())
        print(eval.test_error())

        attack.epochs = 50
        perturbed_probabilities = None
        for b, (inputs, targets) in enumerate(self.testset):
            batchset = [(inputs, targets)]
            perturbed_model = attack.run(self.model, batchset, objective)
            if self.cuda:
                perturbed_model = perturbed_model.cuda()
            perturbed_probabilities = common.numpy.concatenate(perturbed_probabilities, common.test.test(perturbed_model, batchset, cuda=self.cuda))

        eval = common.eval.CleanEvaluation(perturbed_probabilities, self.testset.dataset.labels, validation=0)
        print(eval.test_error())

        attack.normalized = True
        perturbed_probabilities = None
        for b, (inputs, targets) in enumerate(self.testset):
            batchset = [(inputs, targets)]
            perturbed_model = attack.run(self.model, batchset, objective)
            if self.cuda:
                perturbed_model = perturbed_model.cuda()
            perturbed_probabilities = common.numpy.concatenate(perturbed_probabilities,
                                                               common.test.test(perturbed_model, batchset,
                                                                                cuda=self.cuda))

        eval = common.eval.CleanEvaluation(perturbed_probabilities, self.testset.dataset.labels, validation=0)
        print(eval.test_error())


if __name__ == '__main__':
    unittest.main()
