import attacks
import torch.utils.data
import common.torch


class NormalTrainingConfig:
    """
    Configuration for normal training.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.cuda = True
        self.augmentation = None
        self.loss = common.torch.classification_loss
        self.trainloader = None
        self.testloader = None
        self.epochs = None
        self.snapshot = None
        self.finetune = None
        self.fixed_quantization = False
        self.quantization = None
        self.projection = None

        # weight averaging
        self.keep_average = False
        self.keep_average_tau = 0.9975

        # Writer depends on log directory
        self.get_writer = None
        # Optimizer is based on parameters
        self.get_optimizer = None
        # Scheduler is based on optimizer
        self.get_scheduler = None
        # Model is based on data resolution
        self.get_model = None

        self.summary_histograms = False
        self.summary_weights = False
        self.summary_images = False

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.trainloader, torch.utils.data.DataLoader)
        assert len(self.trainloader) > 0
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert self.epochs > 0
        assert self.snapshot is None or self.snapshot > 0
        assert callable(self.get_optimizer)
        assert callable(self.get_scheduler)
        assert callable(self.get_model)
        assert callable(self.get_writer)
        assert self.loss is not None
        assert callable(self.loss)


class AdversarialWeightsTrainingConfig(NormalTrainingConfig):
    """
    Configuration for adversarial training.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(AdversarialWeightsTrainingConfig, self).__init__()

        # Fixed parameters
        self.attack = None
        self.objective = None
        self.operators = None
        self.curriculum = None
        self.average_statistics = False
        self.adversarial_statistics = False
        self.gradient_clipping = 0.05
        self.reset_iterations = 1

    def validate(self):
        """
        Check validity.
        """

        super(AdversarialWeightsTrainingConfig, self).validate()

        assert isinstance(self.attack, attacks.weights.Attack)
        assert isinstance(self.objective, attacks.weights.objectives.Objective)


class AttackWeightsConfig:
    """
    Configuration for attacks.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.directory = None

        # Fixed parameters
        self.trainloader = None
        self.testloader = None
        self.attack = None
        self.objective = None
        self.attempts = None
        self.snapshot = None
        self.eval = True
        self.operators = None
        self.model_specific = False
        self.save_models = False

        # Depends on directory
        self.get_writer = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert isinstance(self.trainloader, torch.utils.data.DataLoader)
        assert isinstance(self.testloader, torch.utils.data.DataLoader)
        assert len(self.testloader) > 0
        assert isinstance(self.attack, attacks.weights.Attack), self.attack
        assert isinstance(self.objective, attacks.weights.objectives.Objective)
        assert callable(self.get_writer)
