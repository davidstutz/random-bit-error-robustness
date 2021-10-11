# README

This repository contains code corresponding to the MLSys'21 paper:

D. Stutz, N. Chandramoorthy, M. Hein, B. Schiele. **On Mitigating Random and Adversarial Bit Errors**. MLSys, 2021.

Please cite as:

    @article{Stutz2020MLSYS,
        author    = {David Stutz and Nandhini Chandramoorthy and Matthias Hein and Bernt Schiele},
        title     = {Bit Error Robustness for Energy-Efficient DNN Acceleratorss},
        booktitle = {Proceedings of Machine Learning and Systems 2021, MLSys 2021},
        publisher = {mlsys.org},
        year      = {2021},
    }

Also check the [project page](https://davidstutz.de/projects/bit-error-training/).

This repository allows to reproduce experiments reported in the paper or use the correspondsing quantization,
weight clipping or training procedures as standalone components.

![Bit Error Robustness for Energy-Efficient DNN Accelerators.](poster.jpg?raw=true "Bit Error Robustness.")

## Overview

* [Installation](#installation)
* [Setup](#setup)
* [Standalone Usage](#standalone-usage)
  * [Bit Errors and Manipulation](#bit-errors-and-manipulation)
  * [Quantization](#quantization)
  * [Random Bit Error Training](#random-bit-error-training)
* [Reproduce Experiments](#reproduce-experiments)
* [License](#license)

## Installation

The following list includes all Python packages required

* torch (including `torch.utils.tensorboard`)
* torchvision
* tensorflow
* tensorboard
* h5py
* json
* numpy
* zipfile
* umap
* sklearn
* imageio
* scipy
* imgaug
* cffi
* cupy

The requirements can be checked using `python3 tests/test_installation.py`. If everything works correctly, all
tests in `tests/` should run without failure. Note that the custom CUDA and C components
in `common/cffi` and `common.cupy.py` require CUDA to be installed (i.e., `nvcc`) and GCC.

Code tested with the following versions:

* Debain 9
* Python 3.5.3
* torch 1.3.1+cu92 (with CUDA 9.2)
* torchvision 0.4.2+cu92
* tensorflow 1.14.0
* tensorboard 1.14.0
* h5py 2.9.0
* numpy 1.18.2
* scipy 1.4.1
* sklearn 0.22.1
* imageio 2.5.0
* imgaug 0.2.9
* cffi 1.13.2
* cupy 7.0.0
* gcc 6.3.0

Also see `environment.yml` for a (not minimal) export of the used environment.

## Download Datasets

To prepare experiments, datasets need to be downloaded and their paths need to be specified:

Check `common/paths.py` and adapt the following variables appropriately:

    # Absolute path to the data directory:
    # BASE_DATA/mnist will contain MNIST
    # BASE_DATA/Cifar10 (capitlization!) will contain Cifar10
    # BASE_DATA/Cifar100 (capitlization!) will contain Cifar100
    BASE_DATA = '/absolute/path/to/data/directory/'
    # Absolute path to experiments directory, experimental results will be written here (i.e., models, perturbed models ...)
    BASE_EXPERIMENTS = '/absolute/path/to/experiments/directory/'
    # Absolute path to log directory (for TensorBoard logs).
    BASE_LOGS = '/absolute/path/to/log/directory/'
    # Absolute path to code directory (this should point to the root directory of this repository)
    BASE_CODE = '/absolute/path/to/root/of/this/repository/'

Download datasets and copy to the appropriate places:

| Dataset | Download |
|---------|----------|
| MNIST | [mnist.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/SYGrssFDciF8st8) |
| CIFAR10 | [cifar10.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/ik2yFHPCyiA74td) |
| CIFAR100 | [cifar100.zip](https://nextcloud.mpi-klsb.mpg.de/index.php/s/rn7Gzpg8brf9GsY) |

### Manual Conversion of Datasets

Download MNIST from the original source [1]. Then, use the scripts in `data` to convert and check the datasets.
For the code to run properly, the datasets are converted to HDF5 format. Cifar is downloaded automatically.

    [1] http://yann.lecun.com/exdb/mnist/

The final dataset directory structure should look as follows:

    BASE_DATE/mnist
    |- t10k-images-idx3-ubyte.gz (downloaded)
    |- t10k-labels-idx-ubyte.gz (downloaded)
    |- train-images-idx3-ubyte.gz (downloaded)
    |- train-labels-idx1-ubyte.gz (downloaded)
    |- train_images.h5 (from data/mnist/convert_mnist.py)
    |- test_images.h5 (from data/mnist/convert_mnist.py)
    |- train_labels.h5 (from data/mnist/convert_mnist.py)
    |- test_labels.h5 (from data/mnist/convert_mnist.py)
    BASE_DATA/Cifar10
    |- cifar-10-batches-py (from torchvision)
    |- cifar-10-python.tar.gz (from torchvision)
    |- train_images.h5 (from data/cifar10/convert_cifar.py)
    |- test_images.h5 (from data/cifar10/convert_cifar.py)
    |- train_labels.h5 (from data/cifar10/convert_cifar.py)
    |- test_labels.h5 (from data/cifar10/convert_cifar.py)
    BASE_DATA/Cifar10
    |- cifar-100-python (from torchvision)
    |- cifar-100-python.tar.gz (from torchvision)
    |- train_images.h5 (from data/cifar100/convert_cifar.py)
    |- test_images.h5 (from data/cifar100/convert_cifar.py)
    |- train_labels.h5 (from data/cifar100/convert_cifar.py)
    |- test_labels.h5 (from data/cifar100/convert_cifar.py)

## Standalone Usage

The code base contains several components that might be interesting to use in a standalone fashion when not
interested in the reproducing the paper's code.

**Overview:** The code base is splitted into several modules:
* `attacks/weights`: containing code to inject bit errors or other perturbations into the weights of a network
* `common`: common utilities
  * `cffi/` and `cupy.py` contain code for bit manipulation on CPU or GPU
  * `eval/` contains evaluation utilities
  * `experiments/` contains experiment utilities
  * `torch/` contains torch utilities, including interfaces to the bit manipulation functions from `cffi/` and `cupy.py`
  * `train/` contains training procedures for normal training and random bit error training
  * `quantization.py` contains quantization implementations
* `experiments/` contains experiment definitions and command line tools, see [Reproduce Experiments](#reproduce-experiments)
* `model/` architecture implementation

### Bit Errors and Manipulation

The code allows direct bit manipulation of torch tensors, both on CPU and GPU. The actual bit manipulation is implemented
in C/CUDA in `common/cffi/` or `common/cupy.py`. The functions are provided in `common.torch`.

Supported data types:
* int32
* int16
* int8
* uint8 (and by extension also quantized tensors with fewer than 8 bits)

Example:

    import common.torch

    # define datatype, int32, int16, int8 or uint8
    dtype = torch.int32
    m = None
    if dtype is torch.int32:
        m = 32
    elif dtype is torch.int16:
        m = 16
    elif dtype is torch.int8 or dtype is torch.uint8:
        m = 8
    else:
        raise ValueError('Invalid dtype selected.')
    
    # use cuda or not
    device = 'cuda'
    
    # print bits
    tensor = torch.zeros([142156], dtype=dtype).to(device)
    bits = common.torch.int_bits(tensor)
    print(tensor, bits)
    
    # hamming distance
    tensor_a = torch.ones([1], dtype=dtype).to(device)
    tensor_b = torch.zeros([1], dtype=dtype).to(device)
    dist = common.torch.int_hamming_distance(tensor_a, tensor_b)
    print('hamming distance between 1 and 0', dist)
    
    # and, or and xor
    tensor_a = torch.tensor([1], dtype=dtype).to(device)
    tensor_b = torch.tensor([3], dtype=dtype).to(device)
    tensor_or = common.torch.int_or(tensor_a, tensor_b)
    tensor_and = common.torch.int_and(tensor_a, tensor_b)
    tensor_xor = common.torch.int_xor(tensor_a, tensor_b)
    print('a', common.torch.int_bits(tensor_a))
    print('b', common.torch.int_bits(tensor_b))
    print('or', common.torch.int_bits(tensor_or))
    print('and', common.torch.int_bits(tensor_and))
    print('xor', common.torch.int_bits(tensor_xor))
    
    # flip and set
    tensor = torch.tensor([1], dtype=torch.int32).to(device)
    mask = [0]*m
    # first element is the least-significant bit
    mask[0] = 1
    mask = torch.tensor([mask]).bool().to(device)
    flipped_tensor = common.torch.int_flip(tensor, mask)
    print('tensor', tensor, common.torch.int_bits(tensor))
    print('set', flipped_tensor, common.torch.int_bits(flipped_tensor))
    
    tensor = torch.tensor([1], dtype=torch.int32).to(device)
    mask1 = [0]*m
    # first element is the least-significant bit
    mask1[1] = 1
    mask1 = torch.tensor([mask1]).bool().to(device)
    mask0 = [0]*m
    mask0[0] = 1
    mask0 = torch.tensor([mask0]).bool().to(device)
    set_tensor = common.torch.int_set(tensor, mask1, mask0)
    print('tensor', tensor, common.torch.int_bits(tensor))
    print('set', set_tensor, common.torch.int_bits(set_tensor))
    
    # random flip
    p = 0.1
    protected_bits = [0]*m
    tensor = torch.randint(0, 100, (1000,), dtype=dtype).to(device)
    flipped_tensor = common.torch.int_random_flip(tensor, p, p, protected_bits)
    dist = common.torch.int_hamming_distance(tensor, flipped_tensor).float().sum()
    print('p', p)
    print('empirical p', dist/(1000*m))

More examples can be found in `tests/test_cffi.py` and `test_cupy.py`.

### Quantization

Various quantization schemes are implemented in `common/quantization.py`. The **main difference to other implementations
is that these methods actually quantize into int32/int16/int8/uint8 instead of just "simulating" this quantization**.
The reason for that is that we need an integer representation to inject bit errors into the quantized weights of
networks.

For 32, 16, we can only quantize into signed integers as torch does not support unsigned 32 or 16 bit integers.
For 8 bit we can also quantize into unsigned integers, which thereby also allows `m`-bit quantization for `m <= 8`.

Additionally, we allow symmetric quantization, with quantization range `[-q_max, q_max]`, and **a**symmetric quantization,
with quantization range `[q_min,q_max]` where `0 < q_min < q_max` or `q_min < q_max < 0` is possible.

In all cases, quantization with fixed `q_max`, adaptive `q_max` per-layer and adaptive `q_max < w_max` with clipping
to `w_max` is provided.

All methods offer the same interface, a `quantize` and a `dequantize` method. In addition to the quantized
tensor, the `quantize` method can return a quantization context (e.g., the adaptive quantization range used)
and the `dequantize` method expects this context as additional argument. Optionally, the `quantize` method can also
be provided with a context if a pre-determined context should be used:

    def quantize(self, tensor, context=None):
        """Quantize tensor."""

        assert isinstance(tensor, torch.Tensor)
        assert context is None or isinstance(context, dict)

    def dequantize(self, tensor, context):
        """De-quantize quantized tensor."""

        assert isinstance(tensor, torch.Tensor)
        assert context is None or isinstance(context, dict)

Quantization into signed integers (32, 16 or 8 bits):
* `AlternativeFixedPointQuantization`
* `AdaptiveAlternativeFixedPointQuantization`
* `ClippedAdaptiveAlternativeFixedPointQuantization`

Quantization into unsigned integers (8 bits or lower down to 2 bits):
* `AlternativeUnsignedFixedPointQuantization`
* `AdaptiveAlternativeUnsignedFixedPointQuantization`
* `ClippedAdaptiveAlternativeUnsignedFixedPointQuantization`

For all six cases, asymmetric variants are available, e.g., `AlternativeUnsymmetricUnsignedFixedPointQuantization`
for `AlternativeUnsignedFixedPointQuantization`.

Example:

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
    
    # ...

`common.quantization.quantize` allows to easily quantize torch models, as well:

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

More examples can be found in `tests/test_cffi.py` and `test_cupy.py`.

Quantization-aware training can be found in `common/train/normal_training.py`.

### Random Bit Error Training

Random bit error training is implemented in various variants in `common/train/`. The main variant used in the paper 
is `common/train/adverage_weights_training.py` which implements the algorithm described in the paper:

![Random Bit Error Training.](training.jpg?raw=true "Random Bit Error Training.")

Implementations of reparameterized group and batch normalization to be used with weight clipping can be found
in `common/train/modules.py`, e.g., `ReparameterizedBatchNorm[1|2]d` and `ReparameterizedGroupNorm`. These can be used
as the default group and batch normalization implementation provided by torch.

## Reproduce Experiments

Experiments are defined in `experiments/mlsys`. For MNIST, CIFAR10 and CIFAR100, the corresponding configuration files,
e.g., `experiments/mlsys/cifar10.py` contain hyper-parameters. The actual models and bit error benchmarks are
defined in `experiments/mlsys/common.py`.

The experiments are run using the command line tools provided in `experiments/`, e.g., `experiments/train.py` for
training a model and `experiments/attack.py` for injecting bit errors. Results are evaluated in Jupyter notebooks,
an examples can be found in `experiments/mlsys/eval/evaluation_cifar10.ipynb`.

All experiments are saved in `BASE_EXPERIMENTS`.

### Training

Training is done using

    cd experiments/mlsys
    python3 train.py mlsys.cifar10 simplenet [model] -n=regn [--whiten]

Models are defined in `experiments/mlsys/common.py` and examples include:

* Training with quantization into unsigned 8-bit integers, with `w_max = 1`: `q81unfp_normal_training`
* Training with clipping, e.g., `w_max = 0.1`: `q801auunrfp_normal_training`
* Random bit error training with `p = 0.01 = 1%` (also `w_max = 0.1`): `q801auunrfp_simple_average_weight_training_bit_random_g001_pop1`

The first part, e.g., `q801auunrfp` is an abbreviation for the number of bits `m` used, the quantization method used
and the used `w_max`, see `experiments/mlsys/common.py` for details. For example, the models can also be trained
using 4 bits: `q41unfp_normal_training` etc.

On CIFAR, training was done with whitening (`--whiten`) and in the paper all models are trained with reparameterized
group normalization (`-n=regn`). However, the code also allows to train ResNets with (reparameterized)
batch normalization instead.

Training automatically takes care of creating snapshots, allowing to restart training if it fails. The models
can then be found in `BASE_EXPERIMENTS/MLSys/Cifar10/model` for, e.g., CIFAR10.

### Random Bit Errors

Injecting random bit errors for evaluation can be done using

    cd experiments/mlsys
    python3 attack.py mlsys.cifar10 simplenet [model] [benchmark] -n=regn [--whiten]

Random bit errors with bit error rate `p = 0.01` are injected using `weight_bit_random_benchmark_g001`. The full suite
of bit error rates, e.g., for CIFAR10 can also be run automatically using `cifar10_benchmark`.

### Evaluation

Evaluation generally happens in Jupyter notebooks as the one in `experiments/mlsys/eval/evaluation_cifar10.ipynb`, for example.
After evaluating the random bit errors above, the predicted probabilities are automatically stored and can be accessed
in `BASE_EXPERIMENTS` and evaluated using `common.eval.AdversarialWeightsEvaluation` (as described above) or following
the code provided in `common.experiments.eval`.

After training the following four models the results obtained using the Jupyter notebook should like as follows:

![Results.](results.png?raw=true "Results.")

* `81unfp_normal_training`
* `q81auunrfp_normal_training`
* `q801auunrfp_normal_training`
* `q801auunrfp_simple_average_weight_training_bit_random_g001_pop1`

## License

This repository includes code from:

* [Coderx7/SimpleNet_Pytorch](https://github.com/Coderx7/SimpleNet_Pytorch)
* [tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape)
* [meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)
* [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
* [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)

Copyright (c) 2021 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.