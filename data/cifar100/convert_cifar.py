import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import paths
from common import utils
from common.log import log
import torchvision
import numpy


def download():
    trainset = torchvision.datasets.CIFAR100(root=paths.raw_cifar100_dir(), train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=paths.raw_cifar100_dir(), train=False, download=True)
    train_images = numpy.array(trainset.train_data)
    train_labels = numpy.array(trainset.train_labels)
    test_images = numpy.array(testset.test_data)
    test_labels = numpy.array(testset.test_labels)

    assert numpy.max(train_images) == 255

    train_images = train_images/255.
    test_images = test_images/255.

    utils.write_hdf5(paths.cifar100_train_images_file(), train_images.astype(numpy.float32))
    log('wrote %s' % paths.cifar100_train_images_file())
    utils.write_hdf5(paths.cifar100_test_images_file(), test_images.astype(numpy.float32))
    log('wrote %s' % paths.cifar100_test_images_file())
    utils.write_hdf5(paths.cifar100_train_labels_file(), train_labels.reshape(-1, 1).astype(numpy.int))
    log('wrote %s' % paths.cifar100_train_labels_file())
    utils.write_hdf5(paths.cifar100_test_labels_file(), test_labels.reshape(-1, 1).astype(numpy.int))
    log('wrote %s' % paths.cifar100_test_labels_file())


if __name__ == '__main__':
    download()