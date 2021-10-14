"""
Initializations for weight attacks.
"""
import common.torch
import torch
import torch.utils.data
import numpy
import common.numpy
from common.log import log


class Initialization:
    """
    Interface for initialization.
    """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        raise NotImplementedError()


class LInfUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon=None, relative_epsilon=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

        self.relative_epsilon = (relative_epsilon is not None)
        """ (bool) Relative epsilon. """

        self.relative_epsilon_fraction = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = float('inf')
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        if self.relative_epsilon:
            max_parameter = 0
            for parameter in model.parameters():
                max_parameter = max(max_parameter, torch.max(torch.abs(parameter)).item())
            self.epsilon = max_parameter*self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        random = self.callable(1, n, epsilon=self.epsilon, ord=self.ord, cuda=cuda).view(-1)

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            perturbed_parameters[i].data = parameters[i].data + random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            n_i += numpy.prod(size_i)


class L2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon=None, relative_epsilon=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

        self.relative_epsilon = (relative_epsilon is not None)
        """ (bool) Relative epsilon. """

        self.relative_epsilon_fraction = relative_epsilon
        """ (float) Relative epsilon fraction. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        if self.relative_epsilon:
            parameters = None
            for parameter in model.parameters():
                parameters = common.numpy.concatenate(parameters, parameter.view(-1).detach().cpu().numpy())
            self.epsilon = numpy.linalg.norm(parameters, ord=self.ord)*self.relative_epsilon_fraction

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        random = self.callable(1, n, epsilon=self.epsilon, ord=self.ord, cuda=cuda).view(-1)

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            perturbed_parameters[i].data = parameters[i].data + random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            n_i += numpy.prod(size_i)


class LayerWiseL2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
    """

    def __init__(self, relative_epsilon, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.relative_epsilon = relative_epsilon
        """ (float) Relative epsilon. """

        self.callable = common.torch.uniform_norm
        """ (callable) Sampler. """

        self.ord = 2
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        cuda = common.torch.is_cuda(model)
        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)

        for i in layers:
            size = list(parameters[i].data.shape)
            epsilon = self.relative_epsilon*torch.norm(parameters[i].data.view(-1), self.ord)
            perturbed_parameters[i].data = parameters[i].data + self.callable(1, numpy.prod(size), epsilon=epsilon, ord=self.ord, cuda=cuda).view(size)


class L0UniformNormInitialization(LInfUniformNormInitialization):
    """
    Uniform initialization, wrt. norm and direction, in L_0 ball.
    """

    def __init__(self, epsilon=None, probability=None, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        assert epsilon is not None or probability is not None
        assert epsilon is None or probability is None

        super(L0UniformNormInitialization, self).__init__(epsilon)

        self.epsilon = epsilon
        """ (float) Epsilon. """

        self.probability = probability
        """ (float= Probability. """

        self.callable = common.utils.partial(common.torch.uniform_norm, low=-1, high=1)
        """ (callable) Sampler. """

        self.ord = 0
        """ (float) Norm. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)
        if self.probability is not None:
            self.epsilon = int(n*self.probability)

        super(L0UniformNormInitialization, self).__call__(model, perturbed_model, layers, quantization=quantization, quantization_contexts=quantization_contexts)


class L0RandomInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, probability, randomness=None):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.probability = probability
        """ (float= Probability. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        # Main reason for overhead is too avoid too many calls to cuda() and rand()!
        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)
        if cuda:
            random = torch.cuda.FloatTensor(n, 2).uniform_(0, 1)
        else:
            random = torch.FloatTensor(n, 2).uniform_(0, 1)

        mask = (random[:, 0] <= self.probability).int()
        random = random[:, 1]*2 - 1 # scale to -1, 1

        n_i = 0
        for i in layers:
            size_i = list(parameters[i].data.shape)
            mask_i = mask[n_i: n_i + numpy.prod(size_i)].view(size_i)
            random_i = random[n_i: n_i + numpy.prod(size_i)].view(size_i)
            perturbed_parameters[i].data = (1 - mask_i)*parameters[i].data + mask_i*random_i
            n_i += numpy.prod(size_i)


class BitRandomInitialization(Initialization):
    """
    Random bit flips.
    """

    def __init__(self, probability, randomness=None):
        """
        Initializer for bit flips.

        :param probability: probability
        :type probability: float
        """

        self.probability = probability
        """ (float) Probability of flip. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0
        assert quantization is not None
        assert quantization_contexts is not None
        assert isinstance(quantization_contexts, list)

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        # Main reason for overhead is too avoid too many calls to cuda() and rand()!
        precision = quantization.type_precision
        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)
        if cuda:
            random = torch.cuda.FloatTensor(n, precision).uniform_(0, 1)
        else:
            random = torch.FloatTensor(n, precision).uniform_(0, 1)
        #log('hash: %s' % hashlib.sha256(random.data.cpu().numpy().tostring()).hexdigest())

        n_i = 0
        for i in layers:
            weights = parameters[i].data
            size_i = list(weights.shape)
            # important: quantization at this point only depends on weights, not on the perturbed weights!
            quantized_weights, _ = quantization.quantize(weights, quantization_contexts[i])
            perturbed_quantized_weights = common.torch.int_random_flip(quantized_weights, self.probability, self.probability,
                                                                       protected_bits=quantization.protected_bits,
                                                                       rand=random[n_i:n_i + numpy.prod(size_i)].view(size_i + [precision]))

            perturbed_dequantized_weights = quantization.dequantize(perturbed_quantized_weights, quantization_contexts[i])
            perturbed_parameters[i].data = perturbed_dequantized_weights
            n_i += numpy.prod(size_i)


class BitRandomMSBInitialization(Initialization):
    """
    Random bit flips.
    """

    def __init__(self, probability, randomness=None):
        """
        Initializer for bit flips.

        :param probability: probability
        :type probability: float
        """

        self.probability = probability
        """ (float) Probability of flip. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0
        assert quantization is not None
        assert quantization_contexts is not None
        assert isinstance(quantization_contexts, list)

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        # Main reason for overhead is too avoid too many calls to cuda() and rand()!
        precision = quantization.type_precision
        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if cuda:
            random = torch.cuda.FloatTensor(n, precision).fill_(1)
        else:
            random = torch.FloatTensor(n, precision).fill_(1)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)
        random[:, quantization.repr_precision - 1].uniform_(0, 1)
        #log('hash: %s' % hashlib.sha256(random.data.cpu().numpy().tostring()).hexdigest())

        n_i = 0
        for i in layers:
            weights = parameters[i].data
            size_i = list(weights.shape)
            # important: quantization at this point only depends on weights, not on the perturbed weights!
            quantized_weights, _ = quantization.quantize(weights, quantization_contexts[i])
            perturbed_quantized_weights = common.torch.int_random_flip(quantized_weights, self.probability, self.probability,
                                                                       protected_bits=quantization.protected_bits,
                                                                       rand=random[n_i: n_i + numpy.prod(size_i)].view(size_i + [precision]))

            # quantized_weights[torch.isnan(quantized_weights)] = 0
            # quantized_weights[torch.isinf(quantized_weights)] = 0

            assert not torch.isnan(quantized_weights).any()
            assert not torch.isinf(quantized_weights).any()

            perturbed_dequantized_weights = quantization.dequantize(perturbed_quantized_weights, quantization_contexts[i])
            perturbed_parameters[i].data = perturbed_dequantized_weights

            n_i += numpy.prod(size_i)


class BitRandomLSBInitialization(Initialization):
    """
    Random bit flips.
    """

    def __init__(self, probability, lsb=15, randomness=None):
        """
        Initializer for bit flips.

        :param probability: probability
        :type probability: float
        """

        self.probability = probability
        """ (float) Probability of flip. """

        self.lsb = lsb
        """ (int) Number of LSBs. """

        self.randomness_ = randomness
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        self.randomness = None
        """ (torch.utils.data.DataLoader) Data loader for random values. """

        if self.randomness_ is not None:
            assert isinstance(self.randomness_, torch.utils.data.DataLoader)
            self.randomness = iter(self.randomness_)

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Initialization.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to initialize
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert len(layers) > 0
        assert quantization is not None
        assert quantization_contexts is not None
        assert isinstance(quantization_contexts, list)

        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())
        assert len(parameters) == len(perturbed_parameters)

        # Main reason for overhead is too avoid too many calls to cuda() and rand()!
        precision = quantization.type_precision
        cuda = common.torch.is_cuda(model)
        n, _, _, _ = common.torch.parameter_sizes(model, layers=None)

        if cuda:
            random = torch.cuda.FloatTensor(n, precision).fill_(1)
        else:
            random = torch.FloatTensor(n, precision).fill_(1)

        if self.randomness is not None:
            try:
                seed = next(self.randomness)[0].item()
            except StopIteration as e:
                log('resetting seeds')
                self.randomness = iter(self.randomness_)
                seed = next(self.randomness)[0].item()
        else:
            seed = numpy.random.randint(0, 2**32 - 1)
        common.torch.torch_numpy_seed(seed)
        random[:, :self.lsb].uniform_(0, 1)
        #log('hash: %s' % hashlib.sha256(random.data.cpu().numpy().tostring()).hexdigest())


        n_i = 0
        for i in layers:
            weights = parameters[i].data
            size_i = list(weights.shape)
            # important: quantization at this point only depends on weights, not on the perturbed weights!
            quantized_weights, _ = quantization.quantize(weights, quantization_contexts[i])
            perturbed_quantized_weights = common.torch.int_random_flip(quantized_weights, self.probability, self.probability,
                                                                       protected_bits=quantization.protected_bits,
                                                                       rand=random[n_i: n_i + numpy.prod(size_i)].view(size_i + [precision]))

            # quantized_weights[torch.isnan(quantized_weights)] = 0
            # quantized_weights[torch.isinf(quantized_weights)] = 0

            assert not torch.isnan(quantized_weights).any()
            assert not torch.isinf(quantized_weights).any()

            perturbed_dequantized_weights = quantization.dequantize(perturbed_quantized_weights, quantization_contexts[i])
            perturbed_parameters[i].data = perturbed_dequantized_weights

            n_i += numpy.prod(size_i)
