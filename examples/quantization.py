import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
import common.quantization

# all example sin 8 bit precision as 8 bit allows both signed and unsigned quantization
precision = 8

# q_max for non-adaptive quantization and w_max for clipped quantization
q_max = 1
w_max = 0.5

# use cuda or not
device = 'cuda'

def quantize_dequantize(quantization, tensor):
    """Simple helper to quantize and dequantize."""
    quantized_tensor, context = quantization.quantize(tensor)
    dequantized_tensor = quantization.dequantize(quantized_tensor, context)
    return quantized_tensor, dequantized_tensor, context

torch.manual_seed(0)
tensor = torch.rand((5)).to(device)*2 - 1
quantization = common.quantization.AlternativeFixedPointQuantization(q_max, precision)
quantized_tensor, dequantized_tensor, _ = quantize_dequantize(quantization, tensor)
print('simple fixed-point quantization (in [-%g, %g]):' % (q_max, q_max))
print('tensor', tensor)
print('quantized_tensor', quantized_tensor)
print('dequantized_tensor', dequantized_tensor)
print('error', torch.abs(dequantized_tensor - tensor))
print()

quantization = common.quantization.AdaptiveAlternativeFixedPointQuantization(precision)
quantized_tensor, dequantized_tensor, _ = quantize_dequantize(quantization, tensor)
print('adaptive fixed-point quantization:')
print('tensor', tensor)
print('quantized_tensor', quantized_tensor)
print('dequantized_tensor', dequantized_tensor)
print('error', torch.abs(dequantized_tensor - tensor))
print()

quantization = common.quantization.ClippedAdaptiveAlternativeFixedPointQuantization(w_max, precision)
quantized_tensor, dequantized_tensor, _ = quantize_dequantize(quantization, tensor)
print('clipped and adaptive fixed-point quantization (clipped to [-%g, %g]:' % (w_max, w_max))
print('tensor', tensor)
print('quantized_tensor', quantized_tensor)
print('dequantized_tensor', dequantized_tensor)
print('error', torch.abs(dequantized_tensor - tensor))
print()

quantization = common.quantization.ClippedAdaptiveAlternativeUnsignedFixedPointQuantization(w_max, precision)
quantized_tensor, dequantized_tensor, _ = quantize_dequantize(quantization, tensor)
print('clipped and adaptive fixed-point quantization into _unsigned_ integers (clipped to [-%g, %g]:' % (w_max, w_max))
print('tensor', tensor)
print('quantized_tensor', quantized_tensor)
print('dequantized_tensor', dequantized_tensor)
print('error', torch.abs(dequantized_tensor - tensor))
print()

quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(w_max, precision)
quantized_tensor, dequantized_tensor, _ = quantize_dequantize(quantization, tensor)
print('asymmetric, clipped and adaptive fixed-point quantization into _unsigned_ integers (clipped to [-%g, %g]:' % (w_max, w_max))
print('tensor', tensor)
print('quantized_tensor', quantized_tensor)
print('dequantized_tensor', dequantized_tensor)
print('error', torch.abs(dequantized_tensor - tensor))
print()


class TestNet(torch.nn.Module):
    """Simple test net for illustration purposes."""
    def __init__(self, D=100, K=10, L=1):
        super(TestNet, self).__init__()
        self.L = L
        for l in range(self.L):
            linear = torch.nn.Linear(D, D)
            torch.nn.init.uniform_(linear.weight, -1, 1)
            setattr(self, 'linear%d' % l, linear)
        self.logits = torch.nn.Linear(D, K)
        torch.nn.init.uniform_(self.logits.weight, -1, 1)

    def forward(self, inputs):
        for l in range(self.L):
            linear = getattr(self, 'linear%d' % l)
            inputs = linear(inputs)
        return self.logits(inputs)


model = TestNet()
layers = ['linear0', 'logits']
print('TestNet:')
for layer in layers:
    module = getattr(model, layer)
    print(layer, 'weights:', module.weight.shape, 'bias:', module.bias.shape)

quantization = common.quantization.ClippedAdaptiveAlternativeUnsymmetricUnsignedFixedPointQuantization(w_max, precision)
dequantized_model, contexts = common.quantization.quantize(quantization, model)

for layer in layers:
    module = getattr(model, layer)
    dequantized_module = getattr(dequantized_model, layer)
    print(layer)
    print('\t', 'weights error:', torch.mean(torch.abs(dequantized_module.weight - module.weight)), 'weights min:',
          torch.min(dequantized_module.weight), 'weights max:', torch.max(dequantized_module.weight))
    print('\t', 'bias error:', torch.mean(torch.abs(dequantized_module.bias - module.bias)), 'bias min:',
          torch.min(dequantized_module.bias), 'bias max:', torch.max(dequantized_module.bias))