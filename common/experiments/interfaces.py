import os
import numpy
import datetime
import common.paths
import common.utils
import common.train
import common.test
import common.eval
import common.state
import attacks.weights.projections
from common.log import log, LogLevel
import attacks
import attacks.weights
from .config import *


def find_incomplete_file(model_file, ext=common.paths.STATE_EXT):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace(ext, '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i]), epochs[i]

    return None, None


def find_incomplete_files(model_file, ext=common.paths.STATE_EXT):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace(ext, '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]
            epochs = sorted(epochs)

            return [os.path.join(base_directory, file_name + '.%d' % epoch) for epoch in epochs], epochs

    return None, None


class NormalTrainingInterface:
    """
    Interface for normal training for expeirments.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, NormalTrainingConfig)
        config.validate()

        self.config = config
        """ (NormalTrainingConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.probabilities_file = None
        """ (str) Probabilities file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.epochs = None
        """ (int) Epochs. """

        self.epoch = None
        """ (int) Start epoch. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.augmentation = None
        """ (None or iaa.meta.Augmenter or torchvision.transforms.Transform) Data augmentation. """

        self.trainloader = None
        """ (torch.utils.data.DataLoader) Train loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.quantization = None
        """ (common.quantization.Quantization) Quantization. """

        self.projection = None
        """ (attacks.weights.projections.Projection) Projection. """

        self.model = None
        """ (torch.nn.Module) Model. """

        self.optimizer = None
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = None
        """ (torch.optim.lr_scheduler.LRScheduler) Scheduler. """

        self.finetune = None
        """ (str) Finetune model. """

        self.summary_histograms = False
        """ (bool) Summary gradients. """

        self.summary_weights = False
        """ (bool) Summary weights. """

    def setup(self):
        """
        Setup.
        """

        dt = datetime.datetime.now()
        self.log_dir = common.paths.log_file(self.config.directory, 'logs/%s' % dt.strftime('%d%m%y%H%M%S'))
        self.model_file = common.paths.experiment_file(self.config.directory, 'classifier', common.paths.STATE_EXT)
        self.probabilities_file = common.paths.experiment_file(self.config.directory, 'probabilities', common.paths.HDF5_EXT)

        self.cuda = self.config.cuda
        self.epochs = self.config.epochs

        self.epoch = 0
        state = None

        self.writer = self.config.get_writer(self.log_dir)
        self.augmentation = self.config.augmentation
        self.loss = self.config.loss
        self.trainloader = self.config.trainloader
        self.testloader = self.config.testloader
        self.quantization = self.config.quantization
        self.projection = self.config.projection
        self.finetune = self.config.finetune
        self.fixed_quantization = self.config.fixed_quantization
        self.summary_histograms = self.config.summary_histograms
        self.summary_weights = self.config.summary_weights
        self.summary_images = self.config.summary_images

        state = self.setup_model()
        self.setup_optimizer(state)

    def setup_model(self):
        incomplete_model_file, epoch = find_incomplete_file(self.model_file)
        load_file = self.model_file
        if os.path.exists(load_file):
            state = common.state.State.load(load_file)
            self.model = state.model
            self.epoch = self.epochs
            log('classifier.pth.tar found, just evaluating', LogLevel.WARNING)
        elif incomplete_model_file is not None and os.path.exists(incomplete_model_file):
            load_file = incomplete_model_file
            state = common.state.State.load(load_file)
            self.model = state.model
            self.epoch = state.epoch + 1
            log('loaded %s, epoch %d' % (load_file, self.epoch))
        else:
            if self.finetune is not None:
                finetune_file = common.paths.experiment_file(self.finetune, 'classifier', common.paths.STATE_EXT)
                probabilities_file = common.paths.experiment_file(self.finetune, 'probabilities', common.paths.HDF5_EXT)
                assert os.path.exists(finetune_file), finetune_file
                assert os.path.exists(probabilities_file)
                state = common.state.State.load(finetune_file)
                self.model = state.model
                if state.epoch is None: # should not happen
                    self.epoch = 100 + 1
                else:
                    self.epoch = state.epoch + 1

                log('fine-tuning %s' % finetune_file)

                probabilities = common.utils.read_hdf5(probabilities_file, 'probabilities')
                eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
                log('fine-tune test error in %%: %g' % (eval.test_error() * 100))
            else:
                state = None
                self.model = self.get_model()
                assert self.model is not None

        if self.cuda:
            self.model = self.model.cuda()

        if self.finetune is not None:
            self.model.eval()
            probabilities = common.test.test(self.model, self.testloader, cuda=self.cuda)
            eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
            log('fine-tune checked test error in %%: %g' % (eval.test_error() * 100))

        print(self.model)
        return state

    def setup_optimizer(self, state):
        self.optimizer = self.config.get_optimizer(self.model)
        if state is not None and self.finetune is None and state.optimizer is not None:
            # fine-tuning should start with fresh optimizer and learning rate
            try:
                self.optimizer.load_state_dict(state.optimizer)
            except ValueError as e:
                log('loaded optimizer dict did not work', LogLevel.WARNING)

        self.scheduler = self.config.get_scheduler(self.optimizer)
        if state is not None and self.finetune is None and state.scheduler is not None:
            self.scheduler.load_state_dict(state.scheduler)
            log('loaded scheduler')

    def get_model(self):
        assert callable(self.config.get_model)
        N_class = numpy.max(self.trainloader.dataset.labels) + 1
        resolution = [
            self.trainloader.dataset.images.shape[3],
            self.trainloader.dataset.images.shape[1],
            self.trainloader.dataset.images.shape[2],
        ]
        model = self.config.get_model(N_class, resolution)
        return model

    def trainer(self):
        """
        Trainer.
        """

        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                              augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)

        return trainer

    def checkpoint(self, model_file, model, epoch=None):
        """
        Save file and check to delete previous file.

        :param model_file: path to file
        :type model_file: str
        :param model: model
        :type model: torch.nn.Module
        :param epoch: epoch of file
        :type epoch: None or int
        """

        if epoch is not None:
            checkpoint_model_file = '%s.%d' % (model_file, epoch)
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)
        else:
            epoch = self.epochs
            checkpoint_model_file = model_file
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)

        previous_model_file = '%s.%d' % (model_file, epoch - 1)
        if os.path.exists(previous_model_file) and (self.config.snapshot is None or (epoch - 1) % self.config.snapshot > 0):
            os.unlink(previous_model_file)

    def main(self):
        """
        Main.
        """

        self.setup()
        trainer = self.trainer()

        if self.config.loss is not None:
            trainer.loss = self.config.loss
        if self.quantization is not None:
            assert isinstance(self.quantization, common.quantization.Quantization)
            trainer.quantization = self.quantization
            trainer.fixed_quantization = self.fixed_quantization
            log('set quantization')
            if trainer.fixed_quantization:
                log('using fixed quantization contexts')
        if self.projection is not None:
            assert isinstance(self.projection, attacks.weights.projections.Projection)
            trainer.projection = self.projection
            log('set projection')

            #max_bound = getattr(trainer.projection, 'max_bound')
            #min_bound = getattr(trainer.projection, 'min_bound')
            #if max_bound is not None:
            #    log('max_bound=%g' % max_bound)
            #if min_bound is not None:
            #    log('min_bound=%g' % min_bound)

        trainer.keep_average = self.config.keep_average
        trainer.keep_average_tau = self.config.keep_average_tau
        log('keep avarage: %r (%g)' % (self.config.keep_average, self.config.keep_average_tau))

        trainer.summary_histograms = self.summary_histograms
        trainer.summary_weights = self.summary_weights
        trainer.summary_images = self.summary_images

        if self.epoch < self.epochs:
            e = self.epochs - 1
            for e in range(self.epoch, self.epochs):
                probabilities, forward_model, contexts = trainer.step(e)
                self.writer.flush()

                self.checkpoint(self.model_file, forward_model, e) # quantized model
                if trainer.quantization is not None:
                    self.checkpoint(self.model_file + 'unquantized', self.model, e) # unquantized model (with floating point updates)

                if trainer.average is not None:
                    self.checkpoint(self.model_file + 'average', trainer.average, e)

                probabilities_file = '%s.%d' % (self.probabilities_file, e)
                common.utils.write_hdf5(probabilities_file, probabilities, 'probabilities')

                previous_probabilities_file = '%s.%d' % (self.probabilities_file, e - 1)
                if os.path.exists(previous_probabilities_file) and (self.config.snapshot is None or (e - 1)%self.config.snapshot > 0):
                    os.unlink(previous_probabilities_file)

            self.checkpoint(self.model_file, forward_model)
            self.checkpoint(self.model_file, forward_model, e)
            if trainer.quantization is not None:
                self.checkpoint(self.model_file + 'unquantized', self.model)
                
            if trainer.average is not None:
                self.checkpoint(self.model_file + 'average', trainer.average)

            previous_probabilities_file = '%s.%d' % (self.probabilities_file, e - 1)
            if os.path.exists(previous_probabilities_file) and (self.config.snapshot is None or (e - 1) % self.config.snapshot > 0):
                os.unlink(previous_probabilities_file)

            forward_model.eval()
            probabilities = common.test.test(forward_model, self.testloader, cuda=self.cuda)
            common.utils.write_hdf5(self.probabilities_file, probabilities, 'probabilities')
        else:
            # when not doing any epochs, self.model is the forward model
            self.model.eval()
            probabilities = common.test.test(self.model, self.testloader, cuda=self.cuda)
            common.utils.write_hdf5(self.probabilities_file, probabilities, 'probabilities')

        eval = common.eval.CleanEvaluation(probabilities, self.testloader.dataset.labels, validation=0)
        log('test error in %%: %g' % (eval.test_error() * 100))


class AdversarialWeightsTrainingInterface(NormalTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        assert isinstance(config, AdversarialWeightsTrainingConfig)

        super(AdversarialWeightsTrainingInterface, self).__init__(config)

        self.trainer_class = common.train.AdversarialWeightsTraining
        """ (class) Trainer class. """

    def trainer(self):
        """
        Trainer.
        """

        if self.quantization is not None:
            self.config.attack.quantization = self.quantization
            log('set attack quantization %s' % self.config.quantization.__class__.__name__)
        if self.projection is not None:
            if self.config.attack.projection is None:
                self.config.attack.projection = self.projection
            else:
                self.config.attack.projection = attacks.weights.projections.SequentialProjections([
                    self.projection,
                    self.config.attack.projection,
                ])
            log('set/added attack projection %s' % self.projection.__class__.__name__)

        trainer = self.trainer_class(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                     self.config.attack, self.config.objective, self.config.operators,
                                     augmentation=self.augmentation, loss=self.loss, writer=self.writer, cuda=self.cuda)
        trainer.curriculum = self.config.curriculum
        trainer.gradient_clipping = self.config.gradient_clipping
        log('gradient clipping %g' % trainer.gradient_clipping)
        trainer.reset_iterations = self.config.reset_iterations
        log('reset_iterations %g' % trainer.reset_iterations)
        if getattr(trainer, 'average_statistics', None) is not None:
            setattr(trainer, 'average_statistics', self.config.average_statistics)
            log('average statistics %g' % getattr(trainer, 'average_statistics', None))
        if getattr(trainer, 'adversarial_statistics', None) is not None:
            setattr(trainer, 'adversarial_statistics', self.config.adversarial_statistics)
            log('adversarial statistics %g' % getattr(trainer, 'adversarial_statistics', None))

        return trainer


class AverageWeightsTrainingInterface(AdversarialWeightsTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        super(AverageWeightsTrainingInterface, self).__init__(config)

        self.trainer_class = common.train.AverageWeightsTraining
        """ (class) Trainer class. """


class AlternatingWeightsTrainingInterface(AdversarialWeightsTrainingInterface):
    """
    Interface for adversarial training.
    """

    def __init__(self, config):
        """
        Initialize.

        :param config: configuration
        :type config: [str]
        """

        super(AlternatingWeightsTrainingInterface, self).__init__(config)

        self.trainer_class = common.train.AlternatingWeightsTraining
        """ (class) Trainer class. """


class AttackWeightsInterface:
    """
    Regular attack interface.
    """

    def __init__(self, target_config, attack_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param attack_config: configuration
        :type attack_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(attack_config, AttackWeightsConfig)

        #target_config.validate()
        attack_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.attack_config = attack_config
        """ (AttackConfig) Config. """

        # Options set in setup
        self.log_dir = None
        """ (str) Log directory. """

        self.model_file = None
        """ (str) Model file. """

        self.perturbations_directory = None
        """ (str) Perturbations directory. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

        self.writer = None
        """ (common.summary.SummaryWriter or torch.utils.tensorboard.SumamryWriter) Summary writer. """

        self.model = None
        """ (torch.nn.Module) Model. """

    def main(self, force_attack=False, force_probabilities=False):
        """
        Main.
        """

        self.log_dir = common.paths.log_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.perturbations_directory = common.paths.experiment_dir('%s/%s' % (self.target_config.directory, self.attack_config.directory))
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        snapshot = self.attack_config.snapshot
        #if os.path.exists(self.model_file):
        #    snapshot = False

        if snapshot is not None:
            self.log_dir = common.paths.log_dir('%s/%s_%d' % (self.target_config.directory, self.attack_config.directory, snapshot))
            self.model_file = self.model_file + '.%d' % snapshot

        assert os.path.exists(self.model_file), 'file %s not found' % self.model_file

        rerun = True
        if os.path.exists(self.perturbations_directory):
            #log('found %s' % self.perturbations_directory)
            rerun = False
            for i in range(self.attack_config.attempts):
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                else:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s' % (i, common.paths.HDF5_EXT))
                if not os.path.exists(probabilities_file):
                    rerun = True
                else:
                    log('found %s' % probabilities_file)

        if rerun or force_attack:
            self.cuda = self.target_config.cuda
            if callable(self.attack_config.get_writer):
                get_writer = common.utils.partial(self.attack_config.get_writer, log_dir=self.log_dir)
                self.writer = get_writer()
            else:
                self.writer = self.attack_config.get_writer

            state = common.state.State.load(self.model_file)
            log('read %s' % self.model_file)
            self.model = state.model

            print(self.model)

            original_quantization = self.attack_config.attack.quantization
            if self.target_config.quantization is not None:
                #assert self.attack_config.attack.quantization is None
                if self.attack_config.attack.quantization is not None:
                    log('overwriting attack quantization', LogLevel.WARNING)
                assert self.attack_config.attack.quantization_contexts is None
                self.attack_config.attack.quantization = self.target_config.quantization
                log('set attack quantization %s' % self.target_config.quantization.__class__.__name__)
                max_abs_range = getattr(self.attack_config.attack.quantization, 'max_abs_range', None)
                if max_abs_range is not None:
                    log('max_abs_range=%g' % max_abs_range)
            if self.target_config.projection is not None:
                if self.attack_config.attack.projection is None:
                    self.attack_config.attack.projection = self.target_config.projection
                else:
                    self.attack_config.attack.projection = attacks.weights.projections.SequentialProjections([
                        self.target_config.projection,
                        self.attack_config.attack.projection,
                    ])
                log('set/added attack projection %s' % self.target_config.projection.__class__.__name__)
                #max_bound = getattr(self.target_config.projection, 'max_bound')
                #min_bound = getattr(self.target_config.projection, 'min_bound')
                #if max_bound is not None:
                #    log('max_bound=%g' % max_bound)
                #if min_bound is not None:
                #    log('min_bound=%g' % min_bound)

            if self.cuda:
                self.model = self.model.cuda()
            if self.attack_config.eval:
                self.model.eval()
            assert self.attack_config.eval is not self.model.training

            perturbed_models = common.test.attack_weights(self.model, self.attack_config.trainloader, self.attack_config.attack,
                                                          self.attack_config.objective, attempts=self.attack_config.attempts,
                                                          writer=self.writer, eval=self.attack_config.eval, cuda=self.cuda)

            for i in range(len(perturbed_models)):
                perturbed_model = perturbed_models[i]

                whiten = getattr(self.model, 'whiten', None)
                if whiten is not None:
                    assert torch.allclose(getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu()), (getattr(perturbed_model, 'whiten').weight.cpu(), whiten.weight.cpu())
                    assert torch.allclose(getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu()), (getattr(perturbed_model, 'whiten').bias.cpu(), whiten.bias.cpu())

                if os.getenv('SAVE_MODELS', None) is not None or self.attack_config.save_models:
                    perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s' % (i, common.paths.STATE_EXT))
                    if snapshot is not None:
                        perturbed_model_file = os.path.join(self.perturbations_directory, 'perturbation%d%s.%d' % (i, common.paths.STATE_EXT, snapshot))
                    common.state.State.checkpoint(perturbed_model_file, perturbed_model)
                    log('saving model file %s!' % perturbed_model_file, LogLevel.WARNING)

                if self.attack_config.eval:
                    perturbed_model.eval()
                assert self.attack_config.eval is not perturbed_model.training
                if self.cuda:
                    perturbed_model = perturbed_model.cuda()

                if self.attack_config.operators is not None:
                    for operator in self.attack_config.operators:
                        operator.reset()

                log('%d/%d' % (i, len(perturbed_models)))
                probabilities = common.test.test(perturbed_model, self.attack_config.testloader, operators=self.attack_config.operators,
                                                 eval=self.attack_config.eval, cuda=self.cuda)
                evaluation = common.eval.CleanEvaluation(probabilities, self.attack_config.testloader.dataset.labels)
                log('error: %g' % evaluation.test_error())
                probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s' % (i, common.paths.HDF5_EXT))
                if snapshot is not None:
                    probabilities_file = os.path.join(self.perturbations_directory, 'probabilities%d%s.%d' % (i, common.paths.HDF5_EXT, snapshot))
                common.utils.write_hdf5(probabilities_file, probabilities, 'probabilities')
                log('wrote %s' % probabilities_file)

            self.attack_config.attack.quantization = original_quantization

        elif force_probabilities:
            raise NotImplementedError()


class QuantizeInterface:
    """
    Regular quantize interface.
    """

    def __init__(self, target_config, quantize_config):
        """
        Initialize.

        :param target_config: configuration
        :type target_config: [str]
        :param quantize_config: configuration
        :type quantize_config: [str]
        """

        assert isinstance(target_config, NormalTrainingConfig)
        assert isinstance(quantize_config, QuantizeConfig)

        #target_config.validate()
        quantize_config.validate()

        self.target_config = target_config
        """ (NormalTrainingConfig) Config. """

        self.quantize_config = quantize_config
        """ (AttackConfig) Config. """

        self.model_file = None
        """ (str) Model file. """

        self.quantized_model_file = None
        """ (str) Model file. """

        self.quantized_probabilities_file = None
        """ (str) Perturbations file. """

        self.cuda = None
        """ (bool) Whether to use CUDA. """

    def main(self, force=False):
        """
        Main.
        """

        self.cuda = self.target_config.cuda
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)
        self.quantized_model_file = common.paths.experiment_file(self.quantize_config.directory, 'classifier', common.paths.STATE_EXT)
        self.quantized_probabilities_file = common.paths.experiment_file(self.quantize_config.directory, 'probabilities', common.paths.HDF5_EXT)

        log(self.quantize_config.directory)
        if force or not os.path.exists(self.quantized_model_file) or not os.path.exists(self.quantized_probabilities_file):
            assert os.path.exists(self.model_file)

            state = common.state.State.load(self.model_file)
            model = state.model
            model.eval()
            if self.cuda:
                model = model.cuda()

            raise NotImplementedError

            quantized_probabilities = common.test.test(model, self.quantize_config.testloader, cuda=self.cuda)
            common.utils.write_hdf5(self.quantized_probabilities_file, quantized_probabilities, 'probabilities')
            log('wrote %s' % self.quantized_probabilities_file)

            evaluation = common.eval.CleanEvaluation(quantized_probabilities, self.quantize_config.testloader.dataset.labels)
            test_error = evaluation.test_error()
            log('test error: %g' % test_error)
        else:
            log('found %s' % self.quantized_model_file)

    def main_attack(self, force=False):
        """
        Main.
        """

        self.cuda = self.target_config.cuda
        self.model_file = common.paths.experiment_file(self.target_config.directory, 'classifier', common.paths.STATE_EXT)

        log(self.quantize_config.directory)
        assert os.path.exists(self.model_file), self.model_file

        for a in range(self.quantize_config.attempts):
            quantized_model_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.quantize_config.directory), 'perturbation%d' % a, ext=common.paths.STATE_EXT)
            if force or not os.path.exists(quantized_model_file):
                state = common.state.State.load(self.model_file)
                model = state.model
                model.eval()
                if self.cuda:
                    model = model.cuda()

                quantized_model, _ = common.quantization.quantize(self.quantize_config.method, model)

                quantized_probabilities = common.test.test(quantized_model, self.quantize_config.testloader, cuda=self.cuda)
                evaluation = common.eval.CleanEvaluation(quantized_probabilities, self.quantize_config.testloader.dataset.labels)
                test_error = evaluation.test_error()
                log('test error: %g' % test_error)

                #quantized_model_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.quantize_config.directory), 'perturbation%d' % a, ext=common.paths.STATE_EXT)
                #common.state.State.checkpoint(quantized_model_file, model)
                #log('wrote %s' % quantized_model_file)

                quantized_probabilities_file = common.paths.experiment_file('%s/%s' % (self.target_config.directory, self.quantize_config.directory), 'probabilities%d' % a, ext=common.paths.HDF5_EXT)
                common.utils.write_hdf5(quantized_probabilities_file, quantized_probabilities, 'probabilities')
                log('wrote %s' % quantized_probabilities_file)
