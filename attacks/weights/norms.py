"""
Norms for weight attacks.
"""
import torch
import common.torch


class Norm:
    def __init__(self):
        """
        Constructor.
        """

        self.norms = []
        """ ([float]) Norms per layer. """

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        raise NotImplementedError()


class LInfNorm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        norm = 0
        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            norm_i = torch.max(torch.abs(perturbation)).item()
            norm = max(norm_i, norm)  # .item() important to avoid GPU memory overhead
            norms.append(norm_i)

        self.norms = norms
        return norm


class RelativeLInfNorm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        norm = 0
        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        max_parameter = 0
        for i in layers:
            max_parameter = max(max_parameter, torch.max(torch.abs(parameters[i].data)).item())
            perturbation = perturbed_parameters[i].data - parameters[i].data
            norm_i = torch.max(torch.abs(perturbation)).item()
            norm = max(norm_i, norm)  # .item() important to avoid GPU memory overhead
            norms.append(norm_i)

        self.norms = norms
        norm /= 2*max_parameter
        return norm


class L2Norm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        perturbations = None
        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            perturbations = common.torch.concatenate(perturbations, perturbation.view(-1))
            norms.append(torch.norm(perturbation, p=2).item())

        self.norms = norms
        return torch.norm(perturbations, p=2).item() # important to avoid GPU memory overhead


class L1Norm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        perturbations = None
        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            perturbations = common.torch.concatenate(perturbations, perturbation.view(-1))
            norms.append(torch.norm(perturbation, p=2).item())

        self.norms = norms
        return torch.norm(perturbations, p=1).item() # important to avoid GPU memory overhead


class L0Norm(Norm):
    def __init__(self, fraction=0.01):
        """
        Constructor.

        :param fraction: fraction of elements to keep in normalization
        :type fraction: float
        """

        assert fraction > 0
        assert fraction <= 1

        self.fraction = fraction

    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        norm = 0
        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            perturbation = perturbed_parameters[i].data - parameters[i].data
            norm_i = torch.norm(perturbation, p=0).item()# important to avoid GPU memory overhead
            norm += norm_i
            norms.append(norm_i)

        self.norms = norms
        return norm


class HammingNorm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert quantization is not None
        #assert quantization_contexts is not None
        assert isinstance(quantization_contexts, list)

        norm = 0
        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            weights = parameters[i].data
            perturbed_weights = perturbed_parameters[i].data

            quantized_weights, _ = quantization.quantize(weights, context=quantization_contexts[i])
            quantized_perturbed_weights, _ = quantization.quantize(perturbed_weights, context=quantization_contexts[i])

            distances = common.torch.int_hamming_distance(quantized_weights, quantized_perturbed_weights)
            norm_i = torch.sum(distances).item()
            norm += norm_i
            norms.append(norm_i)

        self.norms = norms
        return norm


class RelativeHammingNorm(Norm):
    def __call__(self, model, perturbed_model, layers, quantization=None, quantization_contexts=None):
        """
        Norm.

        :param model: original model
        :type model: torch.nn.Module
        :param perturbed_model: perturbed model
        :type perturbed_model: torch.nn.Module
        :param layers: layers to compute norm on
        :type layers: [int]
        :param quantization: quantization if required
        :type quantization: Quantization
        :param quantization_contexts: quantization contexts for each layer
        :type quantization_contexts: [dict]
        """

        assert quantization is not None
        #assert quantization_contexts is not None
        #assert isinstance(quantization_contexts, list)

        norm = 0
        norms = []
        parameters = list(model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for i in layers:
            weights = parameters[i].data
            perturbed_weights = perturbed_parameters[i].data

            quantized_weights, _ = quantization.quantize(weights, context=quantization_contexts[i])
            quantized_perturbed_weights, _ = quantization.quantize(perturbed_weights, context=quantization_contexts[i])

            distances = common.torch.int_hamming_distance(quantized_weights, quantized_perturbed_weights)
            norm_i = torch.sum(distances).item()
            norm += norm_i
            norms.append(norm_i)

        self.norms = norms
        n, _, _, _ = common.torch.parameter_sizes(model)
        norm /= n*quantization.repr_precision
        return norm
