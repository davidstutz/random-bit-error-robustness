"""
https://github.com/Coderx7/SimpleNet_Pytorch
"""

import torch
import common.torch
from .classifier import Classifier
from .utils import get_normalization2d, get_activation


class SimpleNet(Classifier):
    def __init__(self, N_class, resolution=(1, 32, 32), activation='relu', dropout=False, normalization='bn', channels=64, linear=False, pre_pool=False, first_channels=False, skip_last_pool=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param activation: activation function
        :type activation: str
        :param dropout: whether to use dropout
        :type dropout: bool
        """

        assert resolution[1] >= 16 and resolution[2] >= 16
        assert activation in ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'leaky_tanh', 'softsign']

        super(SimpleNet, self).__init__(N_class, resolution, **kwargs)

        self.activation = activation
        """ (str) Activation. """

        activation_layer = get_activation(self.activation)
        assert activation_layer is not None

        self.normalization = normalization
        """ (str) Normalization type. """

        self.dropout = dropout
        """ (str) Dropout. """

        self.channels = channels
        """ (int) Channels. """

        self.linear = linear
        """ (int or bool) Linear layer before logits. """

        self.pre_pool = pre_pool
        """ (int) Pre pool conv layer. """

        self.first_channels = first_channels
        """ (int) First channel size. """

        self.skip_last_pool = skip_last_pool
        """ (bool) Skip one pool, e.g., on Cifar10 to have higher resolution at the end. """

        block = 1
        downsampled = 0

        # 1
        in_channels = resolution[0]
        out_channels = self.channels if self.first_channels is False else self.first_channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 2
        block += 1
        in_channels = self.channels if self.first_channels is False else self.first_channels
        out_channels = 2*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 3
        block += 1
        in_channels = 2*self.channels
        out_channels = 2*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 4
        block += 1
        in_channels = 2*self.channels
        out_channels = 2*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        downsampled += 1
        self.append_layer('pool%d' % block, pool)

        if self.dropout:
            drop = torch.nn.Dropout2d(p=0.1)
            self.append_layer('drop%d' % block, drop)

        # 5
        block += 1
        in_channels = 2*self.channels
        out_channels = 2*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 6
        block += 1
        in_channels = 2*self.channels
        out_channels = 2*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 7
        block += 1
        in_channels = 2*self.channels
        out_channels = 4*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        downsampled += 1
        self.append_layer('pool%d' % block, pool)

        if self.dropout:
            drop = torch.nn.Dropout2d(p=0.1)
            self.append_layer('drop%d' % block, drop)

        # determine whether resolution is enough for extra pool step
        extra_pool = resolution[1] >= 32 and resolution[2] >= 32 and not self.skip_last_pool

        # 8
        block += 1
        # hack for also working on MNIST with 28x28
        in_channels = 4*self.channels
        out_channels = 4*self.channels if extra_pool else 8*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        if extra_pool:
            # 9
            block += 1
            in_channels = 4*self.channels
            out_channels = 4*self.channels
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
            common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(conv.bias, 0)
            self.append_layer('conv%d' % block, conv)
            self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
            relu = activation_layer()
            self.append_layer('relu%d' % block, relu)

            pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            downsampled += 1
            self.append_layer('pool%d' % block, pool)

            if self.dropout:
                drop = torch.nn.Dropout2d(p=0.1)
                self.append_layer('drop%d' % block, drop)

            # 10
            block += 1
            in_channels = 4*self.channels
            out_channels = 8*self.channels
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
            common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(conv.bias, 0)
            self.append_layer('conv%d' % block, conv)
            self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
            relu = activation_layer()
            self.append_layer('relu%d' % block, relu)

            pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            downsampled += 1
            self.append_layer('pool%d' % block, pool)

            if self.dropout:
                drop = torch.nn.Dropout2d(p=0.1)
                self.append_layer('drop%d' % block, drop)

        # 11
        block += 1
        in_channels = 8*self.channels
        out_channels = 32*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        # 12
        block += 1
        in_channels = 32*self.channels
        out_channels = 4*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        downsampled += 1
        self.append_layer('pool%d' % block, pool)

        if self.dropout:
            drop = torch.nn.Dropout2d(p=0.1)
            self.append_layer('drop%d' % block, drop)

        # 13
        block += 1
        in_channels = 4*self.channels
        out_channels = 4*self.channels
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=self.include_bias)
        common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv%d' % block, conv)
        self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
        relu = activation_layer()
        self.append_layer('relu%d' % block, relu)

        if self.pre_pool is not False and self.pre_pool > 0:
            block += 1
            in_channels = 4 * self.channels
            out_channels = self.pre_pool * self.channels
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0), bias=self.include_bias)
            common.torch.kaiming_normal_(conv.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(conv.bias, 0)
            self.append_layer('conv%d' % block, conv)
            self.append_layer('%s%d' % (self.normalization, block), get_normalization2d(self.normalization, out_channels))
            relu = activation_layer()
            self.append_layer('relu%d' % block, relu)

        pool = torch.nn.MaxPool2d(kernel_size=[resolution[1]//(2**downsampled), resolution[2]//(2**downsampled)])
        self.append_layer('pool%d' % block, pool)

        if self.dropout:
            drop = torch.nn.Dropout2d(p=0.1)
            self.append_layer('drop%d' % block, drop)

        view = common.torch.Flatten()
        self.append_layer('view%d' % block, view)

        if self.linear is not False and self.linear > 0:
            block += 1
            linear = torch.nn.Linear(out_channels, self.linear, bias=self.include_bias)
            common.torch.kaiming_normal_(linear.weight, nonlinearity=activation, scale=self.init_scale)
            if self.include_bias:
                torch.nn.init.constant_(linear.bias, 0)
            self.append_layer('linear%d' % block, linear)
            out_channels = self.linear

        logits = torch.nn.Linear(out_channels, self._N_output, bias=self.include_bias)
        common.torch.kaiming_normal_(logits.weight, nonlinearity=activation, scale=self.init_scale)
        if self.include_bias:
            torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def __str__(self):
        """
        Print network.
        """

        string = super(SimpleNet, self).__str__()
        string += '(activation: %s)\n' % self.activation
        string += '(channels: %d)\n' % self.channels
        string += '(normalization: %s)\n' % self.normalization
        string += '(dropout: %s)\n' % self.dropout

        return string