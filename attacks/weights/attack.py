import common.summary
import common.datasets
import random
from .objectives import Objective
from .projections import Projection
from .initializations import Initialization
from common.quantization import Quantization
from common.log import log, LogLevel


class Attack:
    """
    Generic attack on weights.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.writer = None # common.summary.SummaryWriter()
        """ (common.summary.SummaryWriter) Summary writer or Tensorboard writer. """

        self.prefix = ''
        """ (str) Prefix for summary writer. """

        self.progress = None
        """ (common.progress.ProgressBar) Progress bar. """

        self.layers = None
        """ ([int]) Layers. """

        self.get_layers = None
        """ (callable) Get layers to attack. """

        self.initialization = None
        """ (Initialization) Initialization. """

        self.projection = None
        """ (Projection) Projection. """

        self.quantization = None
        """ (Quantization) Quantization. """

        self.quantization_contexts = None
        """ ([dict]) Quantization context for each layer. """

        self.randomize_values = []
        """ (dict) Randomize attack part. """

        self.training = False
        """ (bool) Training mode. """

        self.norm = None
        """ (Norm) Norm. """

        self.auxiliary = False
        """ (bool) Auxiliar layers to be included. """

    def initialize(self, model, perturbed_model):
        """
        Initialize the attack.
        """

        if self.initialization is not None:
            self.initialization(model, perturbed_model, self.layers, self.quantization, self.quantization_contexts)

    def project(self, model, perturbed_model):
        """
        Project the weight perturbation onto the allowed set of perturbations.
        """

        if self.projection is not None:
            self.projection(model, perturbed_model, self.layers, self.quantization, self.quantization_contexts)

    def quantize(self, model, quantized_model=None):
        """
        Quantize the model if necessary.
        """

        if self.quantization is not None:
            return common.quantization.quantize(self.quantization, model, quantized_model, layers=self.layers, contexts=self.quantization_contexts)
        else:
            if quantized_model is not None:
                parameters = list(model.parameters())
                quantized_parameters = list(quantized_model.parameters())
                assert len(parameters) == len(quantized_parameters)

                for i in range(len(parameters)):
                    quantized_parameters[i].data = parameters[i].data

            return common.torch.clone(model), None

    def layers_(self, model):
        """
        Get layers to attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :return: layers
        :rtype: [int]
        """

        if self.get_layers is not None:
            layers = self.get_layers(model)
            named_parameters = dict(model.named_parameters())
            named_parameters_keys = list(named_parameters.keys())
            self.layers = [i for i in layers if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
        else:
            named_parameters = dict(model.named_parameters())
            named_parameters_keys = list(named_parameters.keys())
            self.layers = [i for i in range(len(named_parameters_keys)) if named_parameters[named_parameters_keys[i]].requires_grad is True and (named_parameters_keys[i].find('auxiliary') < 0 or self.auxiliary is True)]
        if 0 in self.layers or 1 in self.layers:
            log('[Warning] layers 0,1 included in attack layers', LogLevel.WARNING)

        return self.layers


    def run(self, model, testset, objective):
        """
        Run the attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: datasets to compute attack on
        :type testset: torch.utils.data.DataLoader
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        """

        assert isinstance(objective, Objective)
        assert self.projection is None or isinstance(self.projection, Projection)
        assert self.initialization is None or isinstance(self.initialization, Initialization)
        assert self.quantization is None or isinstance(self.quantization, Quantization)
        assert self.quantization_contexts is None

        if self.writer is not None:
            self.writer.add_text('%sattack' % self.prefix, self.__class__.__name__)
            self.writer.add_text('%sobjective' % self.prefix, objective.__class__.__name__)

        # Randomize hyper-parameters if requested.
        if len(self.randomize_values) > 0:
            index = random.choice(range(len(self.randomize_values)))
            array = self.randomize_values[index]
            for key in array:
                setattr(self, key, array[key])

        # Set the layers to attack.
        self.layers_(model)

        if self.projection is not None:
            self.projection.reset()