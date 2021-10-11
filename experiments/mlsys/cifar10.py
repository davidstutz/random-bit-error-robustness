import numpy
import common.datasets
import experiments.mlsys.helper as helper

helper.lr = 0.05
helper.epochs = 250
helper.batch_size = 128
helper.base_directory = 'MLSys/Cifar10/'

helper.augmentation = True
helper.mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
helper.cutout = 16
helper.threshold = 1.75

helper.trainset = common.datasets.Cifar10TrainSet()
helper.testset = common.datasets.Cifar10TestSet()
helper.adversarialtrainbatch = common.datasets.Cifar10TestSet(indices=9999 - numpy.array(list(range(100))))
helper.adversarialtrainset = common.datasets.Cifar10TestSet(indices=9999 - numpy.array(list(range(500))))
helper.adversarialtestset = common.datasets.Cifar10TestSet(indices=numpy.array(list(range(9000))))

helper.trainsampler = False

from experiments.mlsys.common import *