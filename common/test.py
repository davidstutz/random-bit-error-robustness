import torch
import numpy
import common.torch
import common.numpy
import common.summary
from common.progress import ProgressBar


def test(model, testset, eval=True, loss=True, operators=None, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    #assert isinstance(testset, torch.utils.data.DataLoader)
    #assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    probabilities = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits = model.forward(inputs, operators=operators)
        # check_nan = torch.sum(logits, dim=1)
        # check_nan = (check_nan != check_nan)
        # if torch.any(check_nan):
        #    log('NaN logits!', LogLevel.WARNING)
        # logits[check_nan, :] = 0.1

        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    #assert probabilities.shape[0] == len(testset.dataset)

    return probabilities


def logits(model, testset, eval=True, loss=True, operators=None, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    #assert isinstance(testset, torch.utils.data.DataLoader)
    #assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    logits = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits_ = model(inputs, operators=operators)
        logits = common.numpy.concatenate(logits, logits_.detach().cpu().numpy())

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits_, targets)
            loss = common.torch.classification_loss(logits_, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    return logits


def features(model, testset, eval=True, loss=True, operators=None, cuda=False, limit=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    #assert isinstance(testset, torch.utils.data.DataLoader)
    #assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    probabilities = None
    features = None

    # should work with and without labels
    for b, data in enumerate(testset):
        if limit is not False and b >= limit:
            break;

        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        logits, features_ = model(inputs, return_features=True, operators=operators)
        #check_nan = torch.sum(logits, dim=1)
        #check_nan = (check_nan != check_nan)
        #if torch.any(check_nan):
        #    log('NaN logits!', LogLevel.WARNING)
        #logits[check_nan, :] = 0.1

        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if features is None:
            features = []
            for i in range(len(features_)):
                features.append(features_[i].reshape(logits.size(0), -1).detach().cpu().numpy())
        else:
            assert len(features) == len(features_)
            for i in range(len(features)):
                features[i] = common.numpy.concatenate(features[i], features_[i].reshape(logits.size(0), - 1).detach().cpu().numpy())

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    #assert probabilities.shape[0] == len(testset.dataset)

    return probabilities, features


def attack_weights(model, testset, attack, objective, attempts=1, start_attempt=0, writer=common.summary.SummaryWriter(), eval=True, cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param writer: summary writer
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    assert attempts > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    if start_attempt > 0:
        initialization = getattr(attack, 'initialization', None)
        if initialization is not None:
            randomness = getattr(initialization, 'randomness', None)
            if randomness is not None:
                for i in range(start_attempt):
                    next(randomness)

    # should work via subsets of datasets
    perturbed_models = []
    for a in range(start_attempt, attempts):
        attack.progress = ProgressBar()
        attack.writer = writer
        attack.prefix = '%d/' % a if not callable(writer) else ''
        objective.reset()
        perturbed_model = attack.run(model, testset, objective)
        if attack.writer is not None:
            attack.writer.flush()
        assert common.torch.is_cuda(perturbed_model) is False
        perturbed_models.append(perturbed_model)

    return perturbed_models
