import numpy
import common.datasets
import experiments.mlsys.helper as helper


helper.lr = 0.05
helper.epochs = 100
helper.batch_size = 128
helper.base_directory = 'MLSys/MNIST/'

helper.augmentation = False
helper.mean = False
helper.cutout = False
helper.threshold = 1.75

helper.trainset = common.datasets.MNISTTrainSet()
helper.testset = common.datasets.MNISTTestSet()
helper.adversarialtrainbatch = common.datasets.MNISTTestSet(indices=9999 - numpy.array(list(range(100))))
helper.adversarialtrainset = common.datasets.MNISTTestSet(indices=9999 - numpy.array(list(range(500))))
helper.adversarialtestset = common.datasets.MNISTTestSet(indices=numpy.array(list(range(9000))))

helper.trainsampler = False

from experiments.mlsys.common import *