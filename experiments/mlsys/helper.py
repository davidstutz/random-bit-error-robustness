lr = None
epochs = None
batch_size = None
base_directory = None

mean = None
cutout = None
threshold = None

trainset = None
testset = None
adversarialtrainbatch = None
adversarialtrainset = None
adversarialtestset = None

trainsampler = None


def guard():
    for key, value in globals().items():
        if not callable(value) and not key.endswith('__'):
            assert value is not None, '%s=%r is None' % (key, value)