"""
Fixed-point quantization schemes.
"""
import torch
import numpy
import common.torch


EPSILON = 0.001


def quantize(quantization, mixed, quantized_mixed=None, contexts=None, layers=None):
    """
    Quantize a model or tensor.

    :param quantization: quantizer
    :type quantization: common.quantization.Quantization
    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.Tensor
    :param quantized_mixed: quantized model or tensor
    :type quantized_mixed: torch.nn.Module or torch.Tensor or None
    :param contexts: context from past quantzation
    :type contexts: mixed
    :return: quantized object
    :rtype: torch.nn.Module or torch.Tensor
    """

    if isinstance(mixed, torch.Tensor):
        if quantized_mixed is None:
            quantized_mixed = torch.zeros_like(mixed)
        quantized_tensor, quantized_context = quantization.quantize(mixed.data, context=contexts)
        quantized_mixed.data = quantization.dequantize(quantized_tensor, quantized_context)
        return quantized_mixed, quantized_context

    elif isinstance(mixed, torch.nn.Module):
        if quantized_mixed is None:
            quantized_mixed = common.torch.clone(mixed)

        parameters = list(mixed.parameters())
        quantized_parameters = list(quantized_mixed.parameters())
        if layers is None:
            layers = list(range(len(parameters)))

        assert contexts is None or len(contexts) == len(parameters)
        if contexts is None:
            contexts = [None] * len(parameters)

        quantized_contexts = []
        for i in range(len(parameters)):
            if i in layers:
                quantized_weights, quantized_context = quantization.quantize(parameters[i].data, context=contexts[i])
                quantized_parameters[i].data = quantization.dequantize(quantized_weights, quantized_context)
                quantized_contexts.append(quantized_context)
            else:
                quantized_parameters[i].data = parameters[i].data
                quantized_contexts.append(None)

        return quantized_mixed, quantized_contexts

    else:
        raise NotImplementedError


def error(original, quantized, ord=2):
    """
    Compute quantization error.

    :param original: original, unquantized object
    :type original: mixed
    :param quantized: quantized objective
    :type quantized: mixed
    :param ord: norm
    :type ord: float
    :return: error
    :rtype: float
    """

    if isinstance(original, torch.Tensor):
        assert isinstance(quantized, torch.Tensor)
        return torch.norm(original - quantized, p=ord).item()

    elif isinstance(original, torch.nn.Module):
        assert isinstance(quantized, torch.nn.Module)

        original_parameters = list(original.parameters())
        quantized_parameters = list(quantized.parameters())
        assert len(original_parameters) == len(quantized_parameters)

        original_weights = None
        quantized_weights = None

        for i in range(len(original_parameters)):
            original_weights = common.torch.concatenate(original_weights, original_parameters[i].view(-1))
            quantized_weights = common.torch.concatenate(quantized_weights, quantized_parameters[i].view(-1))

        return torch.norm(original_weights - quantized_weights, p=ord).item()

    else:
        raise NotImplementedError


class Quantization:
    """
    Simple quantization interface.
    """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        assert isinstance(tensor, torch.Tensor)
        #assert not isinstance(tensor, torch.autograd.Variable) # tensor = autograd now
        assert context is None or isinstance(context, dict)

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        assert isinstance(tensor, torch.Tensor)
        #assert not isinstance(tensor, torch.autograd.Variable)
        assert context is None or isinstance(context, dict)


class AlternativeFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, max_abs_range, precision, double=False, correct=False):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert max_abs_range > 0
        assert precision in [8, 16, 32]

        self.max_abs_range = max_abs_range
        """ (float) maximum absolute value to represent. """

        self.type_precision = precision
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.epsilon = EPSILON
        """ (float) Epsilon. """

        self.delta = (max_abs_range + self.epsilon) / (2**(precision - 1) - 1)
        """ (float) One over decimal range. """

        self.protected_bits = [0] * precision
        """ ([int]) Protected bits. """

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeFixedPointQuantization, self).quantize(tensor, context)
        assert context is None

        quantized_tensor = torch.clamp(tensor, min=-self.max_abs_range, max=self.max_abs_range)
        quantized_tensor = quantized_tensor / self.delta

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 32:
            quantized_tensor = quantized_tensor.to(torch.int32)
        elif self.type_precision == 16:
            quantized_tensor = quantized_tensor.to(torch.int16)
        elif self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.int8)
        else:
            raise NotImplementedError()

        return quantized_tensor, None

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeFixedPointQuantization, self).dequantize(tensor, context)
        assert context is None
        dequantized_tensor = torch.clamp(tensor.to(torch.float32) * self.delta, min=-self.max_abs_range, max=self.max_abs_range)

        return dequantized_tensor


class AdaptiveAlternativeFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert precision in [8, 16, 32]

        self.type_precision = precision
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.epsilon = EPSILON
        """ (float) Minimum range. """

        self.protected_bits = [0] * precision
        """ ([int]) Protected bits. """

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeFixedPointQuantization, self).quantize(tensor, context)

        if context is not None:
            assert isinstance(context, dict)
            assert 'max_abs_range' in context.keys() and 'delta' in context.keys()

        if context is not None:
            max_val = context['max_abs_range']
        else:
            max_val = torch.max(torch.abs(tensor)).item() + self.epsilon

        delta = max_val / (2**(self.repr_precision - 1) - 1)

        if context is not None:
            assert numpy.isclose(delta, context['delta'])

        quantized_tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        quantized_tensor = quantized_tensor / delta

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 32:
            quantized_tensor = quantized_tensor.to(torch.int32)
        elif self.type_precision == 16:
            quantized_tensor = quantized_tensor.to(torch.int16)
        elif self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.int8)
        else:
            raise NotImplementedError()

        context = {
            'max_abs_range': max_val,
            'delta': delta
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_abs_range' in context.keys() and 'delta' in context.keys()
        dequantized_tensor = torch.clamp(tensor.to(torch.float32) * context['delta'], min=-context['max_abs_range'], max=context['max_abs_range'])

        return dequantized_tensor


class ClippedAdaptiveAlternativeFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert precision in [8, 16, 32]

        self.type_precision = precision
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.max_abs_range = max_abs_range
        """ (float) Clip. """

        self.epsilon = EPSILON
        """ (float) Minimum range. """

        self.protected_bits = [0] * precision
        """ ([int]) Protected bits. """

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeFixedPointQuantization, self).quantize(tensor, context)
        if context is not None:
            assert isinstance(context, dict)
            assert 'max_abs_range' in context.keys() and 'delta' in context.keys()
            assert numpy.isclose(context['max_abs_range_'], self.max_abs_range), (context['max_abs_range_'], self.max_abs_range)
            assert context['max_abs_range'] <= self.max_abs_range + 1e-6, (context['max_abs_range'], self.max_abs_range)

        if context is not None:
            max_val = context['max_abs_range']
        else:
            max_val = torch.max(torch.abs(tensor)).item() + self.epsilon

        # ! main difference to adaptive
        max_val = min(max_val, self.max_abs_range)

        delta = max_val / (2 ** (self.repr_precision - 1) - 1)

        if context is not None:
            assert numpy.isclose(delta, context['delta'])

        quantized_tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        quantized_tensor = quantized_tensor / delta

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 32:
            quantized_tensor = quantized_tensor.to(torch.int32)
        elif self.type_precision == 16:
            quantized_tensor = quantized_tensor.to(torch.int16)
        elif self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.int8)
        else:
            raise NotImplementedError()

        context = {
            'max_abs_range_': self.max_abs_range,
            'max_abs_range': max_val,
            'delta': delta,
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_abs_range' in context.keys() and 'delta' in context.keys() and 'max_abs_range_' in context.keys()
        assert numpy.isclose(context['max_abs_range_'], self.max_abs_range), (context['max_abs_range_'], self.max_abs_range)
        dequantized_tensor = torch.clamp(tensor.to(torch.float32) * context['delta'], min=-context['max_abs_range'], max=context['max_abs_range'])

        return dequantized_tensor


class AlternativeUnsymmetricFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeFixedPointQuantization

    def __init__(self, min_range, max_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        self.quantization = self.quantization_class(max_abs_range=1, precision=precision)
        """ (Quantization) Quantization. """

        self.min_range = min_range
        """ (float) Min of quantization range. """

        self.max_range = max_range
        """ (float) Max of quantization range. """

        # avoid overwriting __get_attr__
        self.type_precision = self.quantization.type_precision
        self.repr_precision = self.quantization.repr_precision
        self.protected_bits = self.quantization.protected_bits

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeUnsymmetricFixedPointQuantization, self).quantize(tensor, context)

        quantized_tensor = torch.clamp(tensor, min=self.min_range, max=self.max_range)

        quantized_tensor = (quantized_tensor - self.min_range)/(self.max_range - self.min_range)
        quantized_tensor = quantized_tensor*2 - 1

        quantized_tensor, _ = self.quantization.quantize(quantized_tensor, context=None)
        assert _ is None

        context = None
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeUnsymmetricFixedPointQuantization, self).dequantize(tensor, context)

        dequantized_tensor = self.quantization.dequantize(tensor, context=None)
        dequantized_tensor = (dequantized_tensor + 1) / 2
        dequantized_tensor = dequantized_tensor*(self.max_range - self.min_range) + self.min_range
        dequantized_tensor = torch.clamp(dequantized_tensor, min=self.min_range, max=self.max_range)

        return dequantized_tensor


class AdaptiveAlternativeUnsymmetricFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeFixedPointQuantization

    def __init__(self, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        self.quantization = self.quantization_class(max_abs_range=1, precision=precision)
        """ (Quantization) Quantization. """

        # avoid overwriting __get_attr__
        self.type_precision = self.quantization.type_precision
        self.repr_precision = self.quantization.repr_precision
        self.protected_bits = self.quantization.protected_bits

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeUnsymmetricFixedPointQuantization, self).quantize(tensor, context)

        if context is not None:
            assert isinstance(context, dict)
            assert 'max_val' in context.keys() and 'min_val' in context.keys()

        if context is not None:
            max_val = context['max_val']
            min_val = context['min_val']
        else:
            max_val = torch.max(tensor).item()
            min_val = torch.min(tensor).item()

        quantized_tensor = torch.clamp(tensor, min=min_val, max=max_val)

        quantized_tensor = (quantized_tensor - min_val)/(max_val - min_val)
        quantized_tensor = quantized_tensor*2 - 1

        quantized_tensor, _ = self.quantization.quantize(quantized_tensor, context=None)
        assert _ is None

        context = {
            'max_val': max_val,
            'min_val': min_val
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeUnsymmetricFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_val' in context.keys() and 'min_val' in context.keys()

        max_val = context['max_val']
        min_val = context['min_val']

        dequantized_tensor = self.quantization.dequantize(tensor, context=None)
        dequantized_tensor = (dequantized_tensor + 1) / 2
        dequantized_tensor = dequantized_tensor*(max_val - min_val) + min_val
        dequantized_tensor = torch.clamp(dequantized_tensor, min=min_val, max=max_val)

        return dequantized_tensor


class ClippedAdaptiveAlternativeUnsymmetricFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeFixedPointQuantization

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        self.max_abs_range = max_abs_range
        """ (float) Max abs range. """

        self.quantization = self.quantization_class(max_abs_range=1, precision=precision)
        """ (Quantization) Quantization. """

        # avoid overwriting __get_attr__
        self.type_precision = self.quantization.type_precision
        self.repr_precision = self.quantization.repr_precision
        self.protected_bits = self.quantization.protected_bits

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeUnsymmetricFixedPointQuantization, self).quantize(tensor, context)

        if context is not None:
            assert isinstance(context, dict)
            assert 'max_val' in context.keys() and 'min_val' in context.keys() and 'max_abs_range_' in context.keys()
            assert numpy.isclose(self.max_abs_range, context['max_abs_range_'])

        quantized_tensor = torch.clamp(tensor, min=-self.max_abs_range, max=self.max_abs_range)

        if context is not None:
            max_val = context['max_val']
            min_val = context['min_val']
        else:
            max_val = torch.max(quantized_tensor).item()
            min_val = torch.min(quantized_tensor).item()

        quantized_tensor = torch.clamp(quantized_tensor, min=min_val, max=max_val)

        quantized_tensor = (quantized_tensor - min_val) / (max_val - min_val)
        quantized_tensor = quantized_tensor * 2 - 1

        quantized_tensor, _ = self.quantization.quantize(quantized_tensor, context=None)
        assert _ is None

        context = {
            'max_val': max_val,
            'min_val': min_val,
            'max_abs_range_': self.max_abs_range,
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeUnsymmetricFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_val' in context.keys() and 'min_val' in context.keys() and 'max_abs_range_' in context.keys()
        assert numpy.isclose(self.max_abs_range, context['max_abs_range_'])

        max_val = context['max_val']
        min_val = context['min_val']

        assert abs(max_val) <= self.max_abs_range + 1e-6
        assert abs(min_val) <= self.max_abs_range + 1e-6

        dequantized_tensor = self.quantization.dequantize(tensor, context=None)
        dequantized_tensor = (dequantized_tensor + 1) / 2
        dequantized_tensor = dequantized_tensor * (max_val - min_val) + min_val
        dequantized_tensor = torch.clamp(dequantized_tensor, min=min_val, max=max_val)

        return dequantized_tensor


class AlternativeUnsignedFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert precision >= 2

        self.type_precision = 8
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.max_abs_range = max_abs_range
        """ (float) Max abs range. """

        self.epsilon = EPSILON
        """ (float) Epsilon. """

        self.delta = (self.max_abs_range + self.epsilon) / (2**(self.repr_precision - 1) - 1)
        """ (float) Delta. """

        self.protected_bits = [0] * precision + [1] * (8 - precision)
        """ ([int]) Protected bits. """
        assert len(self.protected_bits) == 8

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeUnsignedFixedPointQuantization, self).quantize(tensor, context)
        assert context is  None

        quantized_tensor = torch.clamp(tensor, min=-self.max_abs_range, max=self.max_abs_range)
        quantized_tensor = quantized_tensor / self.delta

        quantized_tensor += 2 ** (self.repr_precision - 1) # -1
        quantized_tensor = torch.clamp(quantized_tensor, max=2 ** self.repr_precision - 1)

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.uint8)
        else:
            raise NotImplementedError()

        return quantized_tensor, None

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AlternativeUnsignedFixedPointQuantization, self).dequantize(tensor, context)
        assert context is None
        #assert torch.all(tensor <= 2 ** self.repr_precision - 1)

        dequantized_tensor = torch.clamp(tensor.to(torch.float32), min=0, max=2 ** self.repr_precision - 1) - 2**(self.repr_precision - 1) # to shield from unwanted bit flips in the unwanted bits for precision < 8
        dequantized_tensor = torch.clamp(dequantized_tensor * self.delta, min=-self.max_abs_range, max=self.max_abs_range)

        return dequantized_tensor


class AdaptiveAlternativeUnsignedFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert precision >= 2
        assert precision <= 8

        self.type_precision = 8
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.epsilon = EPSILON
        """ (float) Minimum range. """

        self.protected_bits = [0] * precision + [1] * (8 - precision)
        """ ([int]) Protected bits. """

        assert len(self.protected_bits) == 8

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeUnsignedFixedPointQuantization, self).quantize(tensor, context)

        if context is not None:
            assert isinstance(context, dict)
            assert 'max_abs_range' in context.keys() and 'delta' in context.keys()

        if context is not None:
            max_val = context['max_abs_range']
        else:
            max_val = torch.max(torch.abs(tensor)).item() + self.epsilon

        delta = max_val / (2**(self.repr_precision - 1) - 1)

        if context is not None:
            assert numpy.isclose(delta, context['delta'])

        quantized_tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        quantized_tensor = quantized_tensor / delta

        quantized_tensor += 2 ** (self.repr_precision - 1)
        quantized_tensor = torch.clamp(quantized_tensor, max=2 ** self.repr_precision - 1)

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.uint8)
        else:
            raise NotImplementedError()

        context = {
            'max_abs_range': max_val,
            'delta': delta
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(AdaptiveAlternativeUnsignedFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_abs_range' in context.keys() and 'delta' in context.keys()
        #assert torch.all(tensor <= 2 ** self.repr_precision - 1)

        dequantized_tensor = torch.clamp(tensor.to(torch.float32), min=0, max=2 ** self.repr_precision - 1) - 2**(self.repr_precision - 1)# to shield from unwanted bit flips in the unwanted bits for precision < 8
        dequantized_tensor = torch.clamp(dequantized_tensor * context['delta'], min=-context['max_abs_range'], max=context['max_abs_range'])

        return dequantized_tensor


class ClippedAdaptiveAlternativeUnsignedFixedPointQuantization(Quantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        assert precision >= 2

        self.type_precision = 8
        """ (int) Type precision. """

        self.repr_precision = precision
        """ (int) Representation precision. """

        self.max_abs_range = max_abs_range
        """ (float) Clip. """

        self.epsilon = EPSILON
        """ (float) Minimum range. """

        self.protected_bits = [0] * precision + [1] * (8 - precision)
        """ ([int]) Protected bits. """
        assert len(self.protected_bits) == 8

        self.round = None
        """ (callable) Round. """

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeUnsignedFixedPointQuantization, self).quantize(tensor, context)
        if context is not None:
            assert isinstance(context, dict)
            assert 'max_abs_range' in context.keys() and 'delta' in context.keys()

        if context is not None:
            max_val = context['max_abs_range']
        else:
            max_val = torch.max(torch.abs(tensor)).item() + self.epsilon

        # ! main difference to adaptive
        max_val = min(max_val, self.max_abs_range)

        delta = max_val / (2 ** (self.repr_precision - 1) - 1)

        if context is not None:
            assert numpy.isclose(delta, context['delta'])

        quantized_tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        quantized_tensor = quantized_tensor / delta

        quantized_tensor += 2 ** (self.repr_precision - 1)
        # assert torch.all(quantized_tensor <= 2 ** self.repr_precision - 1), quantized_tensor
        quantized_tensor = torch.clamp(quantized_tensor, min=0, max=2 ** self.repr_precision - 1)

        if self.round is not None:
            quantized_tensor = self.round(quantized_tensor)

        if self.type_precision == 8:
            quantized_tensor = quantized_tensor.to(torch.uint8)
        else:
            raise NotImplementedError()

        context = {
            'max_abs_range': max_val,
            'delta': delta
        }
        return quantized_tensor, context

    def dequantize(self, tensor, context):
        """
        De-quantize quantized tensor.

        :param tensor: input quantized tensor
        :type tensor: torch.Tensor
        :param context: additional context such as range
        :type context: mixed
        :return: de-quantized tensor
        :rtype: torch:Tensor
        """

        super(ClippedAdaptiveAlternativeUnsignedFixedPointQuantization, self).dequantize(tensor, context)
        assert isinstance(context, dict)
        assert 'max_abs_range' in context.keys() and 'delta' in context.keys()
        #assert torch.all(tensor <= 2 ** self.repr_precision - 1)

        dequantized_tensor = torch.clamp(tensor.to(torch.float32), min=0, max=2 ** self.repr_precision - 1) - 2**(self.repr_precision - 1)# to shield from unwanted bit flips in the unwanted bits for precision < 8
        dequantized_tensor = torch.clamp(dequantized_tensor * context['delta'], min=-context['max_abs_range'], max=context['max_abs_range'])

        return dequantized_tensor


class ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization(ClippedAdaptiveAlternativeUnsignedFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        super(ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization, self).__init__(max_abs_range, precision)
        self.round = torch.round

    def quantize(self, tensor, context=None):
        """
        Quantize tensor.

        :param tensor: input tensor
        :type tensor: torch.Tensor
        :param context: quantization context;
            allows to use the same context as in past quantization independent of new tensor
        :type context: mixed
        :return: quantized tensor
        :rtype: torch:Tensor
        """

        assert self.round is not None
        return super(ClippedAdaptiveAlternativeUnsignedRoundedFixedPointQuantization, self).quantize(tensor, context)


class AlternativeUnsymmetricUnsignedFixedPointQuantization(AlternativeUnsymmetricFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeUnsignedFixedPointQuantization

class AdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(AdaptiveAlternativeUnsymmetricFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeUnsignedFixedPointQuantization


class ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(ClippedAdaptiveAlternativeUnsymmetricFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeUnsignedFixedPointQuantization


class AdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(AdaptiveAlternativeUnsymmetricFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeUnsignedFixedPointQuantization

    def __init__(self, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        super(AdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization, self).__init__(precision)
        self.quantization.round = torch.round


class ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization(ClippedAdaptiveAlternativeUnsymmetricFixedPointQuantization):
    """
    Simple float to fixed point quantization.
    """

    quantization_class = AlternativeUnsignedFixedPointQuantization

    def __init__(self, max_abs_range, precision):
        """
        Constructor.

        :param precision: precision to quantized in
        :type precision: int
        """

        super(ClippedAdaptiveAlternativeUnsymmetricUnsignedRoundedFixedPointQuantization, self).__init__(max_abs_range, precision)
        self.quantization.round = torch.round