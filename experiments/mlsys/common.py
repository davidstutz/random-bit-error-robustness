import common.experiments
import common.quantization
import common.utils
import common.numpy
import common.paths
import common.autoaugment
from common.log import log, LogLevel
import attacks
import torch
import torch.utils.tensorboard
import torchvision
import numpy
import experiments.mlsys.helper as helper
helper.guard()


def get_training_writer(log_dir, sub_dir=''):
    return common.summary.SummaryWriter()


def get_attack_writer(log_dir, sub_dir=''):
    return common.summary.SummaryWriter()


def get_l2_optimizer(model, lr=helper.lr, momentum=0.9, weight_decay=0.0005):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)


get_default_optimizer = get_l2_optimizer


def get_exponential_scheduler(optimizer, batches_per_epoch, lr_factor=0.9675):
    return common.train.get_exponential_scheduler(optimizer, batches_per_epoch=batches_per_epoch, gamma=lr_factor)


def get_step_scheduler(optimizer, batches_per_epoch, step_size=50, lr_factor=0.1):
    return common.train.get_step_scheduler(optimizer, batches_per_epoch=batches_per_epoch, step_size=step_size, gamma=lr_factor)


def get_multi_step_scheduler(optimizer, batches_per_epoch, milestones=[2*helper.epochs//5, 3*helper.epochs//5, 4*helper.epochs//5], lr_factor=0.1):
    return common.train.get_multi_step_scheduler(optimizer, batches_per_epoch=batches_per_epoch, milestones=milestones, gamma=lr_factor)


get_default_scheduler = get_multi_step_scheduler
finetune_epochs = (helper.epochs + 2*helper.epochs//5)
get_finetune_scheduler = common.utils.partial(get_multi_step_scheduler, milestones=[2*finetune_epochs//5, 3*finetune_epochs//5, 4*finetune_epochs//5])

cuda = True
batch_size = helper.batch_size
epochs = helper.epochs

trainset = helper.trainset
testset = helper.testset
adversarialtrainset = helper.adversarialtrainset
adversarialtrainbatch = helper.adversarialtrainbatch
adversarialtestset = helper.adversarialtestset

if helper.augmentation:
    log('loading data augmentation')
    assert isinstance(helper.mean, list) and len(helper.mean) > 0 and helper.cutout > 0
    data_resolution = trainset.images.shape[1]
    assert trainset.images.shape[1] == trainset.images.shape[2]
    # has to be tensor
    data_mean = torch.tensor(helper.mean)
    # has to be tuple
    data_mean_int = []
    for c in range(data_mean.numel()):
        data_mean_int.append(int(255*data_mean[c]))
    data_mean_int = tuple(data_mean_int)
    trainset.transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda array: (array*255).astype(numpy.uint8)),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(data_resolution, padding=int(data_resolution*0.125), fill=data_mean_int),
        torchvision.transforms.RandomHorizontalFlip(),
        common.autoaugment.CIFAR10Policy(fillcolor=data_mean_int),
        torchvision.transforms.ToTensor(),
        common.torch.CutoutAfterToTensor(n_holes=1, length=helper.cutout, fill_color=data_mean),
        torchvision.transforms.Lambda(lambda array: array.permute(1, 2, 0)),
    ])
else:
    log('[Warning] no data augmentation', LogLevel.WARNING)

trainsampler = helper.trainsampler
if not trainsampler:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
adversarialtrainsetloader = torch.utils.data.DataLoader(adversarialtrainset, batch_size=batch_size, shuffle=False)
adversarialtrainbatchloader = torch.utils.data.DataLoader(adversarialtrainbatch, batch_size=batch_size, shuffle=False)
adversarialtestloader = torch.utils.data.DataLoader(adversarialtestset, batch_size=batch_size, shuffle=False)


def normal_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer', 'quantization',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'lr', 'fixed_quantization', 'loss',
    ]))


    config = common.experiments.NormalTrainingConfig()
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', None)
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.quantization = kwargs.get('quantization', None)
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.fixed_quantization = kwargs.get('fixed_quantization', False)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 25)
    config.directory = '%s/%s' % (helper.base_directory, directory)
    config.interface = common.experiments.NormalTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def adversarial_weights_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer', 'quantization',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'attack', 'objective', 'curriculum',
        'lr', 'fixed_quantization', 'loss',
        'operators',
    ]))

    config = common.experiments.AdversarialWeightsTrainingConfig()
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.operators = kwargs.get('operators', None)
    config.curriculum = kwargs.get('curriculum', None)
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', None)
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.quantization = kwargs.get('quantization', None)
    config.projection = kwargs.get('projection', None)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.fixed_quantization = kwargs.get('fixed_quantization', False)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 25)
    config.directory = '%s/%s' % (helper.base_directory, directory)

    config.interface = common.experiments.AdversarialWeightsTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def average_weights_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer', 'quantization',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'attack', 'objective', 'curriculum',
        'lr', 'average_statistics', 'adversarial_statistics',
        'fixed_quantization', 'loss',
        'operators',
    ]))

    config = common.experiments.AdversarialWeightsTrainingConfig()
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.operators = kwargs.get('operators', None)
    config.curriculum = kwargs.get('curriculum', None)
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', None)
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.quantization = kwargs.get('quantization', None)
    config.projection = kwargs.get('projection', None)
    config.average_statistics = kwargs.get('average_statistics', False)
    config.adversarial_statistics = kwargs.get('adversarial_statistics', False)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.fixed_quantization = kwargs.get('fixed_quantization', False)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 25)
    config.directory = '%s/%s' % (helper.base_directory, directory)

    config.interface = common.experiments.AverageWeightsTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def alternating_weights_training_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'cuda', 'augmentation', 'trainloader', 'testloader',
        'epochs', 'get_writer', 'get_optimizer', 'quantization',
        'projection', 'finetune', 'get_scheduler', 'snapshot',
        'attack', 'objective', 'curriculum',
        'lr', 'average_statistics', 'adversarial_statistics',
        'fixed_quantization', 'loss',
        'operators#,'
    ]))

    config = common.experiments.AdversarialWeightsTrainingConfig()
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.operators = kwargs.get('operators', None)
    config.curriculum = kwargs.get('curriculum', None)
    config.cuda = kwargs.get('cuda', cuda)
    config.augmentation = kwargs.get('augmentation', None)
    config.trainloader = kwargs.get('trainloader', trainloader)
    config.testloader = kwargs.get('testloader', testloader)
    config.epochs = kwargs.get('epochs', epochs)
    config.get_writer = kwargs.get('get_writer', get_training_writer)
    config.get_optimizer = kwargs.get('get_optimizer', common.utils.partial(get_default_optimizer, lr=kwargs.get('lr', helper.lr)))
    config.quantization = kwargs.get('quantization', None)
    config.projection = kwargs.get('projection', None)
    config.average_statistics = kwargs.get('average_statistics', False)
    config.adversarial_statistics = kwargs.get('adversarial_statistics', False)
    config.loss = kwargs.get('loss', common.torch.classification_loss)
    config.finetune = kwargs.get('finetune', None)
    config.fixed_quantization = kwargs.get('fixed_quantization', False)
    config.get_scheduler = kwargs.get('get_scheduler', common.utils.partial(get_default_scheduler, batches_per_epoch=len(config.trainloader)))
    config.snapshot = kwargs.get('snapshot', 25)
    config.directory = '%s/%s' % (helper.base_directory, directory)

    config.interface = common.experiments.AlternatingWeightsTrainingInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def attack_weights_config(directory, **kwargs):
    """
    Set a variable as a global variable with given name.

    :param directory: name of global
    :type directory: str
    """

    keys = set(kwargs.keys())
    assert keys.issubset(set([
        'trainloader', 'testloader', 'attack', 'objective', 'attempts', 'get_writer', 'eval',
        'operators',
    ]))

    config = common.experiments.AttackWeightsConfig()
    config.trainloader = kwargs.get('trainloader', None)
    assert config.trainloader is not None
    config.testloader = kwargs.get('testloader', adversarialtestloader)
    assert config.testloader is not None
    config.attack = kwargs.get('attack', None)
    assert config.attack is not None
    config.objective = kwargs.get('objective', attacks.weights.objectives.UntargetedF0Objective())
    assert config.objective is not None
    config.attempts = kwargs.get('attempts', 1)
    config.eval = kwargs.get('eval', True)
    config.operators = kwargs.get('operators', None)
    config.get_writer = kwargs.get('get_writer', get_attack_writer)
    config.directory = directory
    config.interface = common.experiments.AttackWeightsInterface

    assert directory not in globals().keys(), directory
    globals()[directory] = config


def gformat(value):
    """
    Format value for directory name.

    :param value: value
    :type value: float
    :return: str
    :rtype: str
    """

    return ('%.7f' % float(value)).rstrip('0').replace('.', '')


max_precision = 8
max_parameters = 6000000
max_attempts = 50
max_layers = 100
general_error_rates = [
    0.2, 0.175, 0.15, 0.125, 0.16, 0.14, 0.12, 0.1,
    0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01,
    0.009, 0.0075, 0.005, 0.0045, 0.0025, 0.001,
    0.00075, 0.0005, 0.00025, 0.0001,
    0.000075, 0.00005, 0.000025, 0.00001,
    0.0000075, 0.000005, 0.0000025,
]

bit_randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(100, 100 + max_attempts)))))
bit_randomloader = torch.utils.data.DataLoader(bit_randomset) 

bit_stress_randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(100, 100 + 100000)))))
bit_stress_randomloader = torch.utils.data.DataLoader(bit_stress_randomset)

bit_input_randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy.array(list(range(100, 100 + 9000*50)))))
bit_input_randomloader = torch.utils.data.DataLoader(bit_input_randomset)

numpy_bit_activation_random = numpy.array(list(range(100, 100 + max_attempts*max_layers)))
numpy_bit_activation_random = numpy_bit_activation_random.reshape(max_layers, max_attempts).T
assert numpy_bit_activation_random.shape[0] == max_attempts
bit_activation_randomset = torch.utils.data.TensorDataset(torch.from_numpy(numpy_bit_activation_random))
bit_activation_randomloader = torch.utils.data.DataLoader(bit_activation_randomset)

#
#
# attacks
#
#

attempts = 50
for general_error_rate in general_error_rates:
    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=attempts)

    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate, randomness=bit_randomloader)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_benchmark_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=attempts)

    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate, randomness=bit_randomloader)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_train_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=attempts, eval=False)

    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate, randomness=bit_stress_randomloader)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_stress_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=500)
    for t in range(1, 200):
        attack = attacks.weights.RandomAttack()
        attack.epochs = 1
        attack.norm = attacks.weights.norms.HammingNorm()
        attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate, randomness=bit_stress_randomloader)
        attack.projection = None  # set by model/quantization
        attack.quantization = None  # set by model
        attack_weights_config('weight_bit_random_stress_g%s_%d' % (gformat(general_error_rate), t),
                              trainloader=adversarialtrainbatchloader,
                              attack=attack, attempts=500)

    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomMSBInitialization(general_error_rate)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_msb_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=attempts)

    attack = attacks.weights.RandomAttack()
    attack.epochs = 1
    attack.norm = attacks.weights.norms.HammingNorm()
    attack.initialization = attacks.weights.initializations.BitRandomMSBInitialization(general_error_rate, randomness=bit_randomloader)
    attack.projection = None  # set by model/quantization
    attack.quantization = None  # set by model
    attack_weights_config('weight_bit_random_msb_benchmark_g%s' % gformat(general_error_rate),
                          trainloader=adversarialtrainbatchloader,
                          attack=attack, attempts=attempts)

log('set up attacks ...')

#
#
# training
#
#

def adversarial_simple_curriculum(attack, loss, perturbed_loss, epoch, threshold=helper.threshold):
    max_loss = max(loss, perturbed_loss)
    if max_loss > threshold:
        population = 0
    else:
        population = 1
    return population, {
        'population': population,
        'epochs': attack.epochs,
    }


def average_simple_curriculum(attack, loss, perturbed_loss, epoch, threshold=helper.threshold):
    if loss < threshold*1.15:
        average = (loss + perturbed_loss)/2
        if average > threshold:
            population = 0
        else:
            population = 1
    else:
        population = 0

    return population, {
        'population': population,
        'epochs': attack.epochs,
    }


def average_step_curriculum(attack, loss, perturbed_loss, epoch, start_epoch=0, epochs=epochs, p=0.01, steps=20, threshold=helper.threshold):
    if loss < threshold*1.15:
        average = (loss + perturbed_loss) / 2
        if average > threshold:
            population = 0
        else:
            population = 1
    else:
        population = 0

    for i in range(steps):
        if epoch >= start_epoch + i*((epochs//2)//steps):
            attack.initialization.probability = (i + 1)*(p/steps)

    return population, {
        'population': population,
        'epochs': attack.epochs,
        'probability': attack.initialization.probability,
    }


def simple_curriculum(attack, loss, perturbed_loss, epoch, threshold=helper.threshold):
    if loss > threshold:
        population = 0
    else:
        population = 1

    return population, {
        'population': population,
        'epochs': attack.epochs,
    }


clippings = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1]
for clipped in clippings:
    projection = attacks.weights.projections.BoxProjection(-clipped, clipped)
    assert projection is not None

    for precision, quantization_class, projected in zip([
        2, 3, 4, 8,
        2, 3, 4, 8,
        2, 3, 4, 8,
        2, 3, 4, 8,
        2, 3, 4, 8,
    ], [
        common.quantization.AlternativeUnsignedFixedPointQuantization,
        common.quantization.AlternativeUnsignedFixedPointQuantization,
        common.quantization.AlternativeUnsignedFixedPointQuantization,
        common.quantization.AlternativeUnsignedFixedPointQuantization,
        #
        common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization,
        #
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization,
        #
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization,
        #
        common.quantization.ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization,
        common.quantization.ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization,
    ], [
        'unfp', 'unfp', 'unfp', 'unfp',
        'aunfp', 'aunfp', 'aunfp', 'aunfp',
        'auunfp', 'auunfp', 'auunfp', 'auunfp',
        'auunrfp', 'auunrfp', 'auunrfp', 'auunrfp',
        'aunrfp', 'aunrfp', 'aunrfp', 'aunrfp',
    ]):

        if projected == 'unfp':
            assert type(quantization_class(max_abs_range=clipped, precision=precision)) == common.quantization.AlternativeUnsignedFixedPointQuantization
        elif projected == 'aunfp':
            assert type(quantization_class(max_abs_range=clipped, precision=precision)) == common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization
        elif projected == 'auunfp':
            assert type(quantization_class(max_abs_range=clipped, precision=precision)) == common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization
        elif projected == 'auunrfp':
            assert type(quantization_class(max_abs_range=clipped, precision=precision)) == common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization
        elif projected == 'aunrfp':
            assert type(quantization_class(max_abs_range=clipped, precision=precision)) == common.quantization.ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization
        else:
            assert False

        quantization = quantization_class(max_abs_range=clipped, precision=precision)
        projection = attacks.weights.projections.BoxProjection(-clipped, clipped)

        assert projection is not None
        assert quantization is not None

        normal_training_config(
            'q%d%s%s_normal_training' % (precision, gformat(clipped), projected),
            quantization=quantization,
            projection=projection,
        )

        for epsilon in [0.01, 0.05, 0.1]:
            normal_training_config(
                'q%d%s%s_normal_training_ls%s' % (precision, gformat(clipped), projected, gformat(epsilon)),
                quantization=quantization,
                projection=projection,
                loss=common.utils.partial(common.torch.smooth_classification_loss, epsilon=epsilon, K=10),
            )

        for general_error_rate in general_error_rates:
            attack = attacks.weights.RandomAttack()
            attack.epochs = 1
            attack.norm = attacks.weights.norms.HammingNorm()
            attack.initialization = attacks.weights.initializations.BitRandomInitialization(general_error_rate)
            attack.projection = None # set by model/quantization
            attack.quantization = None # set by model

            adversarial_weights_training_config(
                'q%d%s%s_simple_adversarial_weight_training_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=adversarial_simple_curriculum,
                quantization=quantization,
                projection=projection,
            )
            alternating_weights_training_config(
                'q%d%s%s_simple_alternating_weight_training_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=average_simple_curriculum,
                quantization=quantization,
                projection=projection,
            )
            average_weights_training_config(
                'q%d%s%s_simple_average_weight_training_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=average_simple_curriculum,
                quantization=quantization,
                projection=projection,
            )
            average_weights_training_config(
                'q%d%s%s_step_average_weight_training_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=common.utils.partial(average_step_curriculum, p=general_error_rate),
                quantization=quantization,
                projection=projection,
            )
            alternating_weights_training_config(
                'q%d%s%s_step_alternating_weight_training_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=common.utils.partial(average_step_curriculum, p=general_error_rate),
                quantization=quantization,
                projection=projection,
            )
            average_weights_training_config(
                'q%d%s%s_simple_average_weight_training_finetune_fixed_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=average_simple_curriculum,
                quantization=quantization,
                fixed_quantization=True,
                projection=projection,
                finetune='%s/q%d%s%s_normal_training' % (helper.base_directory, precision, gformat(clipped), projected),
                epochs=epochs + int(2*epochs/5.),
                lr=helper.lr/10,
                get_scheduler=common.utils.partial(get_finetune_scheduler, batches_per_epoch=len(trainloader)),
            )
            average_weights_training_config(
                'q%d%s%s_step_average_weight_training_finetune_fixed_bit_random_g%s_pop1' % (precision, gformat(clipped), projected, gformat(general_error_rate)),
                attack=attack,
                curriculum=common.utils.partial(average_step_curriculum, start_epoch=epochs, epochs=epochs + int(2 * epochs / 5.), p=general_error_rate, steps=10),
                quantization=quantization,
                fixed_quantization=True,
                projection=projection,
                finetune='%s/q%d%s%s_normal_training' % (helper.base_directory, precision, gformat(clipped), projected),
                epochs=epochs + int(2 * epochs / 5.),
                lr=helper.lr / 10,
                get_scheduler=common.utils.partial(get_finetune_scheduler, batches_per_epoch=len(trainloader)),
            )

log('set up models ...')

cifar10_benchmark = [
    globals()['weight_bit_random_benchmark_g000005'],
    globals()['weight_bit_random_benchmark_g00001'],
    globals()['weight_bit_random_benchmark_g000025'],
    globals()['weight_bit_random_benchmark_g00005'],
    globals()['weight_bit_random_benchmark_g000075'],
    globals()['weight_bit_random_benchmark_g0001'],
    globals()['weight_bit_random_benchmark_g00025'],
    globals()['weight_bit_random_benchmark_g0005'],
    globals()['weight_bit_random_benchmark_g00075'],
    globals()['weight_bit_random_benchmark_g001'],
    globals()['weight_bit_random_benchmark_g0015'],
    globals()['weight_bit_random_benchmark_g002'],
    globals()['weight_bit_random_benchmark_g0025'],
]
mnist_benchmark = [
    globals()['weight_bit_random_benchmark_g0001'],
    globals()['weight_bit_random_benchmark_g0005'],
    globals()['weight_bit_random_benchmark_g001'],
    globals()['weight_bit_random_benchmark_g002'],
    globals()['weight_bit_random_benchmark_g003'],
    globals()['weight_bit_random_benchmark_g004'],
    globals()['weight_bit_random_benchmark_g005'],
    globals()['weight_bit_random_benchmark_g01'],
    globals()['weight_bit_random_benchmark_g015'],
    globals()['weight_bit_random_benchmark_g02'],
]
cifar100_benchmark = [
    globals()['weight_bit_random_benchmark_g00000025'],
    globals()['weight_bit_random_benchmark_g0000005'],
    globals()['weight_bit_random_benchmark_g00000075'],
    globals()['weight_bit_random_benchmark_g000001'],
    globals()['weight_bit_random_benchmark_g0000025'],
    globals()['weight_bit_random_benchmark_g000005'],
    globals()['weight_bit_random_benchmark_g0000075'],
    globals()['weight_bit_random_benchmark_g00001'],
    globals()['weight_bit_random_benchmark_g000025'],
    globals()['weight_bit_random_benchmark_g00005'],
    globals()['weight_bit_random_benchmark_g000075'],
    globals()['weight_bit_random_benchmark_g0001'],
    globals()['weight_bit_random_benchmark_g00025'],
    globals()['weight_bit_random_benchmark_g0005'],
    globals()['weight_bit_random_benchmark_g00075'],
    globals()['weight_bit_random_benchmark_g001'],
]