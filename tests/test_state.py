import unittest
import torch
import numpy
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import models
import common.state


import torch
import numpy


class TestNet(models.Classifier):
    """
    LeNet classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        """

        super(TestNet, self).__init__()

        self.resolution = resolution
        self.N_class = N_class
        self.kwargs = kwargs
        self.__layers = []

        logits = torch.nn.Linear(numpy.prod(resolution), self.N_class)
        self.append_layer('logits', logits)

    def append_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.append(name)

    def forward(self, image):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :return: logits
        :rtype: torch.autograd.Variable
        """

        output = image
        for name in self.__layers:
            output = getattr(self, name)(output)
        return output


class TestState(unittest.TestCase):
    def setUp(self):
        self.filepath = 'test.pth.tar'
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)

    def testSimple(self):

        net = TestNet(10)
        print(net)
        state = common.state.State(net)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model


    def testModels(self):
        model_classes = [
            'LeNet',
            'MLP',
            'ResNet'
        ]

        for model_class in model_classes:
            model_class = common.utils.get_class('models', model_class)
            original_model = model_class(10, [1, 32, 32])
            for parameters in original_model.parameters():
                parameters.data.zero_()

            state = common.state.State(original_model)
            state.save(self.filepath)

            state = common.state.State.load(self.filepath)
            loaded_model = state.model

            self.assertEqual(loaded_model.__class__.__name__, original_model.__class__.__name__)
            self.assertListEqual(loaded_model.resolution, original_model.resolution)

            for parameters in loaded_model.parameters():
                self.assertEqual(torch.sum(parameters).item(), 0)

    def testModelOnly(self):
        original_model = models.LeNet(10, [1, 32, 32])
        for parameters in original_model.parameters():
            parameters.data.zero_()

        state = common.state.State(original_model)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model

        self.assertEqual(loaded_model.__class__.__name__, original_model.__class__.__name__)
        self.assertListEqual(loaded_model.resolution, original_model.resolution)

        for parameters in loaded_model.parameters():
            self.assertEqual(torch.sum(parameters).item(), 0)

    def testModelOptimizer(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        state = common.state.State(original_model, original_optimizer)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)

        for param_group in loaded_optimizer.param_groups:
            self.assertEqual(param_group['lr'], 0.01)
            self.assertEqual(param_group['momentum'], 0.9)

    def testModelOptimizerScheduler(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        state = common.state.State(original_model, original_optimizer, original_scheduler)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)
        loaded_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        loaded_scheduler.load_state_dict(state.scheduler)

        self.assertEqual(original_scheduler.step_size, loaded_scheduler.step_size)
        self.assertEqual(original_scheduler.gamma, loaded_scheduler.gamma)

    def testModelOptimizerSchedulerEpoch(self):
        original_model = models.LeNet(10, [1, 32, 32])
        original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
        original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        original_epoch = 100
        state = common.state.State(original_model, original_optimizer, original_scheduler, original_epoch)
        state.save(self.filepath)

        state = common.state.State.load(self.filepath)
        loaded_model = state.model
        loaded_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.99, momentum=0.1)
        loaded_optimizer.load_state_dict(state.optimizer)
        loaded_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=10, gamma=0.9)
        loaded_scheduler.load_state_dict(state.scheduler)
        loaded_epoch = state.epoch

        self.assertEqual(original_epoch, loaded_epoch)

    def testResNet(self):
        resolutions = [
            [3, 32, 32],
        ]

        blocks = [
            [3],
            [3, 3, 3, 3],
        ]

        block_types = [
            '',
            'bottleneck',
        ]

        normalizations = [
            '',
            'bn',
        ]

        clamps = [
            True,
            False
        ]

        scales_and_whitens = [
            (False, True),
            (False, False),
            (True, False),
        ]

        classes = 10
        for resolution in resolutions:
            for block in blocks:
                for block_type in block_types:
                    for normalization in normalizations:
                        for clamp in clamps:
                            for scale_and_whiten in scales_and_whitens:
                                print(resolution, blocks, block_type, normalization, clamp, scale_and_whiten)
                                original_model = models.ResNet(classes, resolution, clamp=clamp, scale=scale_and_whiten[0],
                                                               whiten=scale_and_whiten[1], blocks=block, block=block_type, normalization=normalization)
                                for parameters in original_model.parameters():
                                    parameters.data.zero_()

                                common.state.State.checkpoint(self.filepath, original_model)
                                state = common.state.State.load(self.filepath)
                                loaded_model = state.model

                                for parameters in loaded_model.parameters():
                                    self.assertEqual(torch.sum(parameters).item(), 0)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)


if __name__ == '__main__':
    unittest.main()
