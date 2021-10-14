import os
import common.experiments
import common.utils
import common.eval
import common.paths
import common.imgaug
import common.datasets
import common.plot
import common.summary
from common.log import log, LogLevel
import numpy
import terminaltables
import matplotlib
matplotlib.rcParams['figure.dpi'] = 125
from matplotlib import pyplot as plt
from IPython.display import display, Markdown


class CheapAttackConfig:
    def __init__(self, config):
        self.directory = config.directory
        self.attempts = config.attempts


class CheapTrainingConfig:
    def __init__(self, config):
        self.directory = config.directory
        self.epochs = config.epochs


def get_log_directory(config, training_config):
    return common.paths.log_dir(training_config.directory)


def get_model_file(config, training_config):
    model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
    model_epochs = training_config.epochs
    if not os.path.exists(model_file):
        model_file, model_epochs = common.experiments.find_incomplete_file(model_file)

    return model_file, model_epochs


def get_model_files(config, training_config):
    model_file = common.paths.experiment_file(training_config.directory, 'classifier', common.paths.STATE_EXT)
    model_files, model_epochs = common.experiments.find_incomplete_files(model_file)
    assert model_files is not None and model_epochs is not None
    model_epochs.append(None)
    model_files.append(model_file)

    return model_files, model_epochs


def get_probabilities_file(config, training_config, train=False, epoch=None):
    if train:
        probabilities_file = common.paths.experiment_file(training_config.directory, 'train_probabilities', common.paths.HDF5_EXT)
    else:
        probabilities_file = common.paths.experiment_file(training_config.directory, 'probabilities', common.paths.HDF5_EXT)

    if epoch is None:
        probabilities_epochs = training_config.epochs
        if not os.path.exists(probabilities_file):
            probabilities_file, probabilities_epochs = common.experiments.find_incomplete_file(probabilities_file)
    else:
        probabilities_epochs = epoch
        probabilities_file += '.%d' % epoch
        if not os.path.exists(probabilities_file):
            probabilities_file = None

    return probabilities_file, probabilities_epochs


def get_perturbed_model_file(config, training_config, attack_config, attempt):
    perturbed_model_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbation%d' % attempt, common.paths.STATE_EXT)
    perturbed_model_epochs = training_config.epochs

    if not os.path.exists(perturbed_model_file):
        perturbed_model_file, perturbed_model_epochs = common.experiments.find_incomplete_file(perturbed_model_file)

    if perturbed_model_file is not None:
        if common.utils.creation_date(perturbed_model_file) < 1600880416.4045188:
            log('[Warning] too old: %s' % perturbed_model_file)

    return perturbed_model_file, perturbed_model_epochs

def get_perturbed_probabilities_file(config, training_config, attack_config, attempt, basename='probabilities', epoch=None):
    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), '%s%d' % (basename, attempt), common.paths.HDF5_EXT)

    if epoch is None:
        adversarial_probabilities_epochs = training_config.epochs
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file, adversarial_probabilities_epochs = common.experiments.find_incomplete_file(adversarial_probabilities_file)
    else:
        adversarial_probabilities_epochs = epoch
        adversarial_probabilities_file += '.%d' % epoch
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file = None

    return adversarial_probabilities_file, adversarial_probabilities_epochs


def get_perturbations_file(config, training_config, attack_config, epoch=None):
    adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations', common.paths.HDF5_EXT)

    if epoch is None:
        adversarial_probabilities_epochs = training_config.epochs
        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file, adversarial_probabilities_epochs = common.experiments.find_incomplete_file(adversarial_probabilities_file)
    else:
        adversarial_probabilities_file = common.paths.experiment_file('%s/%s' % (training_config.directory, attack_config.directory), 'perturbations%d' % epoch, common.paths.HDF5_EXT)
        adversarial_probabilities_epochs = epoch

        if not os.path.exists(adversarial_probabilities_file):
            adversarial_probabilities_file = None

    return adversarial_probabilities_file, adversarial_probabilities_epochs


def load(config, training_config_vars, training_suffixes, attack_config_vars):
    attack_configs = []
    for attack in attack_config_vars:
        if isinstance(attack, list):
            attack_configs_ = []
            for attack_ in attack:
                attack_configs_.append(CheapAttackConfig(getattr(config, attack_)))
            attack_configs.append(attack_configs_)
        else:
            attack_configs.append(CheapAttackConfig(getattr(config, attack)))
    training_configs = []
    for t in range(len(training_config_vars)):
        training_config = CheapTrainingConfig(getattr(config, training_config_vars[t]))
        if isinstance(training_suffixes, list):
            training_config.directory += training_suffixes[t]
        else:
            training_config.directory += training_suffixes
        training_configs.append(training_config)

    return training_configs, attack_configs


def get_attack_evaluations(config, training_configs, attack_configs, evaluation_class=common.eval.AdversarialWeightsEvaluation, limit=9000, debug=False, train=False, epoch=None):
    evaluations = []
    epochs = []
    epoch_table = []

    testset = config.testloader.dataset
    if train:
        testset = config.testtrainloader.dataset

    for training_config in training_configs:
        attack_evaluations = []
        attack_epochs = []
        for attack_config in attack_configs:
            evaluation = None
            evaluation_epoch = None
            if isinstance(attack_config, list):
                if training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)

                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        attempt_evaluations = []
                        min_adversarial_probabilities_epoch = 1e12
                        for attack_config_ in attack_config:
                            if attack_config_ is not None:
                                # find adversarial probabilities
                                a_count = 0

                                for a in range(attack_config_.attempts):
                                    adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config_, a, epoch=epoch)

                                    if adversarial_probabilities_file is not None:
                                        adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                        if adversarial_probabilities.shape[0] < limit:
                                            pass
                                        else:
                                            adversarial_probabilities = adversarial_probabilities[:limit]
                                        adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                        attempt_evaluations.append(adversarial_evaluation)

                                        min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                        a_count += 1
                                epoch_table.append([attack_config_.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = min_adversarial_probabilities_epoch
            else:
                if attack_config is not None and training_config is not None:
                    probabilities_file, probabilities_epochs = get_probabilities_file(config, training_config, train=train, epoch=epoch)
                    if probabilities_file is not None:
                        clean_probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                        epoch_table.append(['**' + training_config.directory + '**', probabilities_epochs, ''])

                        a_count = 0
                        min_adversarial_probabilities_epoch = 1e12

                        attempt_evaluations = []
                        for a in range(attack_config.attempts):
                            adversarial_probabilities_file, adversarial_probabilities_epochs = get_perturbed_probabilities_file(config, training_config, attack_config, a, epoch=epoch)

                            if adversarial_probabilities_file is not None:
                                adversarial_probabilities = common.utils.read_hdf5(adversarial_probabilities_file, 'probabilities')
                                if adversarial_probabilities.shape[0] < limit:
                                    pass
                                else:
                                    adversarial_probabilities = adversarial_probabilities[:limit]
                                adversarial_evaluation = evaluation_class(clean_probabilities, adversarial_probabilities, testset.labels)
                                attempt_evaluations.append(adversarial_evaluation)

                                min_adversarial_probabilities_epoch = min(min_adversarial_probabilities_epoch, adversarial_probabilities_epochs)
                                a_count += 1

                        if len(attempt_evaluations) > 0:
                            evaluation = common.eval.EvaluationStatistics(attempt_evaluations)
                            evaluation_epoch = adversarial_probabilities_epochs

                        epoch_table.append([attack_config.directory, str(min_adversarial_probabilities_epoch), str(a_count)])

            attack_evaluations.append(evaluation)
            attack_epochs.append(evaluation_epoch)
        evaluations.append(attack_evaluations)
        epochs.append(attack_epochs)

    if debug:
        table = terminaltables.GithubFlavoredMarkdownTable(epoch_table)
        display(Markdown(table.table))
    return evaluations, epochs


def plot_attack_evaluations_line(training_configs, attack_configs, evaluations, metric='robust_test_error', statistic='mean',
                                 metric_at_zero=None, reference=None, threshold=None, training_labels=None, latex_labels=None, attack_labels=None,
                                 **kwargs):
    if training_labels is not None:
        assert len(training_labels) == len(training_configs)
    if attack_labels is not None:
        assert len(attack_labels) == len(attack_configs)
        for i in range(len(attack_labels)):
            attack_labels[i] = float(attack_labels[i])

    labels = []

    xs = []
    ys = []
    yerrls = []
    yerrhs = []

    for t in range(len(training_configs)):
        training_config = training_configs[t]
        labels.append(training_config.directory)

        x = []
        y = []
        yerrl = []
        yerrh = []

        if metric_at_zero is not None:
            a = 0
            while a < len(attack_configs) - 1 and evaluations[t][a] is None:
                a += 1
            if evaluations[t][a] is not None:
                x.append(0)
                y.append(round(evaluations[t][a](metric_at_zero, 'mean')[0], 4) * 100)
                yerrl.append(0)
                yerrh.append(0)

        if attack_labels is not None:
            x += attack_labels
        else:
            x += list(range(len(attack_configs)))

        for a in range(len(attack_configs)):
            if t == 0:
                attack_config = attack_configs[a]
                if isinstance(attack_config, list):
                    print('x = %d:' % a)
                    for attack_config_ in attack_config:
                        print('\t%s' % attack_config_.directory)
                else:
                    print('x = %d: %s' % (a, attack_config.directory))

            if evaluations[t][a] is not None:
                if isinstance(evaluations[t][a], common.eval.EvaluationStatistics):
                    value = round(evaluations[t][a](metric, statistic)[0], 4) * 100
                    value_min = round(evaluations[t][a](metric, 'min')[0], 4) * 100
                    value_max = round(evaluations[t][a](metric, 'max')[0], 4) * 100

                    y.append(value)
                    yerrl.append(abs(value - value_min))
                    yerrh.append(abs(value - value_max))
                else:
                    value = round(getattr(evaluations[t][a], metric)(), 4) * 100
                    y.append(value)
                    yerrl.append(0)
                    yerrh.append(0)
            else:
                y.append(0)
                yerrl.append(0)
                yerrh.append(0)

        xs.append(numpy.array(x))
        ys.append(numpy.array(y))
        yerrls.append(numpy.array(yerrl))
        yerrhs.append(numpy.array(yerrh))

    if training_labels is not None:
        labels = training_labels

    if threshold is not None:
        labels.insert(0, 'Threshold')
        if metric_at_zero is not None:
            x = [0]
            y = [threshold]
            yerrl = [0]
            yerrh = [0]
        else:
            x = []
            y = []
            yerrl = []
            yerrh = []

        x += attack_labels if attack_labels is not None else list(range(len(attack_configs)))
        y += [threshold] * len(attack_configs)
        yerrl += [0] * len(attack_configs)
        yerrh += [0] * len(attack_configs)

        xs.insert(0, numpy.array(x))
        ys.insert(0, numpy.array(y))
        yerrls.insert(0, numpy.array(yerrl))
        yerrhs.insert(0, numpy.array(yerrh))

    if reference is not None:
        labels.insert(0, 'Reference')
        if metric_at_zero is not None:
            x = [0]
            y = [reference]
            yerrl = [0]
            yerrh = [0]
        else:
            x = []
            y = []
            yerrl = []
            yerrh = []

        x += attack_labels if attack_labels is not None else list(range(len(attack_configs)))
        y += [reference] * len(attack_configs)
        yerrl += [0] * len(attack_configs)
        yerrh += [0] * len(attack_configs)

        xs.insert(0, numpy.array(x))
        ys.insert(0, numpy.array(y))
        yerrls.insert(0, numpy.array(yerrl))
        yerrhs.insert(0, numpy.array(yerrh))

    common.plot.errorbar(xs, ys, yerrl=yerrls, yerrh=yerrhs, labels=labels, ax=plt.gca(), **kwargs)
    plt.show()
