import numpy
import common.datasets
import experiments.mlsys.helper as helper


helper.lr = 0.05
helper.epochs = 250
helper.batch_size = 128
helper.base_directory = 'MLSys/Cifar100/'

helper.augmentation = True
helper.mean = [0.50707459, 0.48654896, 0.44091788]
helper.cutout = 16
helper.threshold = 3.5

helper.trainset = common.datasets.Cifar100TrainSet()
helper.testset = common.datasets.Cifar100TestSet()
helper.adversarialtrainbatch = common.datasets.Cifar100TestSet(indices=9999 - numpy.array(list(range(100))))
helper.adversarialtrainset = common.datasets.Cifar100TestSet(indices=9999 - numpy.array(list(range(500))))
helper.adversarialtestset = common.datasets.Cifar100TestSet(indices=numpy.array(list(range(9000))))

helper.trainsampler = False

from experiments.mlsys.common import *