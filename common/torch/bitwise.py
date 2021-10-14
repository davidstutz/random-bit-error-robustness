"""
Bitwise tensor manipulation.
"""
import torch
import math
import common.cffi as cffi
import common.cupy as cupy
from .utils import is_cuda, topk


def check_type(*tensors):
    def check_precision(tensor):
        if tensor.dtype == torch.int32:
            return 32
        elif tensor.dtype == torch.int16:
            return 16
        elif tensor.dtype == torch.int8:
            return 8
        elif tensor.dtype == torch.uint8:
            return 8
        else:
            raise NotImplementedError

    precision = None
    for tensor in tensors:
        assert (tensor.dtype == torch.int32) or (tensor.dtype == torch.int16) or (tensor.dtype == torch.int8) or (tensor.dtype == torch.uint8)
        if precision is None:
            precision = check_precision(tensor)
        else:
            assert precision == check_precision(tensor), 'tensor should be %d-bit, but is %d-bit' % (precision, check_precision(tensor))

    return precision


def int_bitwise_operation(a, b, name):
    """
    Bit-wise operation between float tensors.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :param name: name of cupy kernel
    :type name: str
    :return: bit-wise and
    :rtype: torch.Tensor
    """

    #assert (a.is_contiguous() == True)
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    check_type(a, b)
    cuda = is_cuda(a)
    assert is_cuda(b) is cuda

    assert len(a.shape) == len(a.shape), (a.shape, b.shape)
    for d in range(len(a.shape)):
        assert a.shape[d] == b.shape[d], (a.shape, b.shape, d)

    c = a.new_zeros(a.shape)
    n = c.nelement()
    shape = list(c.shape)
    grid, block = cupy.grid_block(shape)

    type = str(a.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%s%s' % (type, name))(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  a.data_ptr(),
                  b.data_ptr(),
                  c.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)

        if name == 'and':
            if type == 'int32':
                _a = cffi.ffi.cast('int*', a.data_ptr())
                _b = cffi.ffi.cast('int*', b.data_ptr())
                _c = cffi.ffi.cast('int*', c.data_ptr())
                cffi.lib.cffi_int32and(_n, _a, _b, _c)
            elif type == 'int16':
                _a = cffi.ffi.cast('short*', a.data_ptr())
                _b = cffi.ffi.cast('short*', b.data_ptr())
                _c = cffi.ffi.cast('short*', c.data_ptr())
                cffi.lib.cffi_int16and(_n, _a, _b, _c)
            elif type == 'int8':
                _a = cffi.ffi.cast('char*', a.data_ptr())
                _b = cffi.ffi.cast('char*', b.data_ptr())
                _c = cffi.ffi.cast('char*', c.data_ptr())
                cffi.lib.cffi_int8and(_n, _a, _b, _c)
            elif type == 'uint8':
                _a = cffi.ffi.cast('unsigned char*', a.data_ptr())
                _b = cffi.ffi.cast('unsigned char*', b.data_ptr())
                _c = cffi.ffi.cast('unsigned char*', c.data_ptr())
                cffi.lib.cffi_uint8and(_n, _a, _b, _c)
            else:
                raise NotImplementedError
        elif name == 'or':
            if type == 'int32':
                _a = cffi.ffi.cast('int*', a.data_ptr())
                _b = cffi.ffi.cast('int*', b.data_ptr())
                _c = cffi.ffi.cast('int*', c.data_ptr())
                cffi.lib.cffi_int32or(_n, _a, _b, _c)
            elif type == 'int16':
                _a = cffi.ffi.cast('short*', a.data_ptr())
                _b = cffi.ffi.cast('short*', b.data_ptr())
                _c = cffi.ffi.cast('short*', c.data_ptr())
                cffi.lib.cffi_int16or(_n, _a, _b, _c)
            elif type == 'int8':
                _a = cffi.ffi.cast('char*', a.data_ptr())
                _b = cffi.ffi.cast('char*', b.data_ptr())
                _c = cffi.ffi.cast('char*', c.data_ptr())
                cffi.lib.cffi_int8or(_n, _a, _b, _c)
            elif type == 'uint8':
                _a = cffi.ffi.cast('unsigned char*', a.data_ptr())
                _b = cffi.ffi.cast('unsigned char*', b.data_ptr())
                _c = cffi.ffi.cast('unsigned char*', c.data_ptr())
                cffi.lib.cffi_uint8or(_n, _a, _b, _c)
            else:
                raise NotImplementedError
        elif name == 'xor':
            if type == 'int32':
                _a = cffi.ffi.cast('int*', a.data_ptr())
                _b = cffi.ffi.cast('int*', b.data_ptr())
                _c = cffi.ffi.cast('int*', c.data_ptr())
                cffi.lib.cffi_int32xor(_n, _a, _b, _c)
            elif type == 'int16':
                _a = cffi.ffi.cast('short*', a.data_ptr())
                _b = cffi.ffi.cast('short*', b.data_ptr())
                _c = cffi.ffi.cast('short*', c.data_ptr())
                cffi.lib.cffi_int16xor(_n, _a, _b, _c)
            elif type == 'int8':
                _a = cffi.ffi.cast('char*', a.data_ptr())
                _b = cffi.ffi.cast('char*', b.data_ptr())
                _c = cffi.ffi.cast('char*', c.data_ptr())
                cffi.lib.cffi_int8xor(_n, _a, _b, _c)
            elif type == 'uint8':
                _a = cffi.ffi.cast('unsigned char*', a.data_ptr())
                _b = cffi.ffi.cast('unsigned char*', b.data_ptr())
                _c = cffi.ffi.cast('unsigned char*', c.data_ptr())
                cffi.lib.cffi_uint8xor(_n, _a, _b, _c)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError()

    return c


def int_and(a, b):
    """
    Bit-wise and between float tensors.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :return: bit-wise and
    :rtype: torch.Tensor
    """

    return int_bitwise_operation(a, b, 'and')


def int_or(a, b):
    """
    Bit-wise and between float tensors.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :return: bit-wise and
    :rtype: torch.Tensor
    """

    return int_bitwise_operation(a, b, 'or')


def int_xor(a, b):
    """
    Bit-wise and between float tensors.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :return: bit-wise and
    :rtype: torch.Tensor
    """

    return int_bitwise_operation(a, b, 'xor')


def int_msb_projection(original, perturbed):
    """
    Bit-wise and between float tensors.

    :param original: original tensor
    :type original: torch.Tensor
    :param perturbed: perturbed tensor
    :type perturbed: torch.Tensor
    :return: msb projection of perturbed onto original
    :rtype: torch.Tensor
    """

    if not original.is_contiguous():
        original = original.contiguous()
    if not perturbed.is_contiguous():
        perturbed = perturbed.contiguous()

    check_type(original, perturbed)
    cuda = is_cuda(original)
    assert is_cuda(perturbed) is cuda

    assert len(original.shape) == len(perturbed.shape), (original.shape, perturbed.shape)
    for d in range(len(original.shape)):
        assert original.shape[d] == perturbed.shape[d], (original.shape, perturbed.shape, d)

    output = torch.zeros_like(original)
    n = original.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)

    type = str(original.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%smsbprojection' % type)(
            # https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  original.data_ptr(),
                  perturbed.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)

        if type == 'int32':
            _original = cffi.ffi.cast('int*', original.data_ptr())
            _perturbed = cffi.ffi.cast('int*', perturbed.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32msbprojection(_n, _original, _perturbed, _output)
        elif type == 'int16':
            _original = cffi.ffi.cast('short*', original.data_ptr())
            _perturbed = cffi.ffi.cast('short*', perturbed.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16msbprojection(_n, _original, _perturbed, _output)
        elif type == 'int8':
            _original = cffi.ffi.cast('char*', original.data_ptr())
            _perturbed = cffi.ffi.cast('char*', perturbed.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8msbprojection(_n, _original, _perturbed, _output)
        elif type == 'uint8':
            _original = cffi.ffi.cast('unsigned char*', original.data_ptr())
            _perturbed = cffi.ffi.cast('unsigned char*', perturbed.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8msbprojection(_n, _original, _perturbed, _output)
        else:
            raise NotImplementedError

    return output


def int_hamming_projection(original, perturbed, epsilon, method='topk'):
    """
    Hamming projection with an additional hamming constraint of 1 per word.

    :param original: original tensor
    :type original: torch.Tensor
    :param perturbed: perturbed tensor
    :type perturbed: torch.Tensor
    :param epsilon: epsilon
    :type epsilon: int
    :param method: method to use
    :type method: str
    :return: msb projection of perturbed onto original
    :rtype: torch.Tensor
    """

    if method == 'sort':
        return int_hamming_projection_sort(original, perturbed, epsilon)
    elif method == 'topk':
        return int_hamming_projection_topk(original, perturbed, epsilon)
    else:
        raise NotImplementedError


def int_hamming_projection_topk(original, perturbed, epsilon):
    """
    Hamming projection with an additional hamming constraint of 1 per word.

    :param original: original tensor
    :type original: torch.Tensor
    :param perturbed: perturbed tensor
    :type perturbed: torch.Tensor
    :param epsilon: epsilon
    :type epsilon: int
    :return: msb projection of perturbed onto original
    :rtype: torch.Tensor
    """

    # assert epsilon >= 0
    # https://stackoverflow.com/questions/51433741/rearranging-a-3-d-array-using-indices-from-sorting
    # size = list(tensor.shape)
    # sorted, indices = torch.sort(tensor.view(tensor.size()[0], -1), dim=1, descending=True)
    # k = int(math.ceil(epsilon))
    # assert k > 0
    # sorted[:, min(k, sorted.size(1) - 1):] = 0
    # print(sorted)
    # tensor = tensor.scatter_(dim=1, index=indices, src=sorted)
    # tensor = tensor.view(size)

    if not original.is_contiguous():
        original = original.contiguous()
    if not perturbed.is_contiguous():
        perturbed = perturbed.contiguous()

    check_type(original, perturbed)
    cuda = is_cuda(original)
    assert is_cuda(perturbed) is cuda

    assert len(original.shape) == len(original.shape), (original.shape, perturbed.shape)
    for d in range(len(original.shape)):
        assert original.shape[d] == perturbed.shape[d], (original.shape, perturbed.shape, d)

    size = original.shape
    original = original.view(-1)
    perturbed = perturbed.view(-1)
    assert epsilon >= 0, epsilon

    perturbed_float = torch.clone(perturbed).float()
    original_float = torch.clone(original).float()
    difference = torch.abs(perturbed_float - original_float)
    k = int(math.ceil(epsilon))
    k = min(k, difference.size(0) - 1)
    assert k > 0, k
    _, top_indices = topk(difference, k=k)

    projection_top_original = original[top_indices]
    projection_top_perturbed = perturbed[top_indices]
    projection_top_projected = int_msb_projection(projection_top_original, projection_top_perturbed)
    #projection_top_projected = projection_top_perturbed

    projected = torch.clone(original)
    projected[top_indices] = projection_top_projected

    return projected.view(size)


def int_hamming_projection_sort(original, perturbed, epsilon):
    """
    Hamming projection with an additional hamming constraint of 1 per word.

    :param original: original tensor
    :type original: torch.Tensor
    :param perturbed: perturbed tensor
    :type perturbed: torch.Tensor
    :param epsilon: epsilon
    :type epsilon: int
    :return: msb projection of perturbed onto original
    :rtype: torch.Tensor
    """

    # assert epsilon >= 0
    # https://stackoverflow.com/questions/51433741/rearranging-a-3-d-array-using-indices-from-sorting
    # size = list(tensor.shape)
    # sorted, indices = torch.sort(tensor.view(tensor.size()[0], -1), dim=1, descending=True)
    # k = int(math.ceil(epsilon))
    # assert k > 0
    # sorted[:, min(k, sorted.size(1) - 1):] = 0
    # print(sorted)
    # tensor = tensor.scatter_(dim=1, index=indices, src=sorted)
    # tensor = tensor.view(size)

    if not original.is_contiguous():
        original = original.contiguous()
    if not perturbed.is_contiguous():
        perturbed = perturbed.contiguous()

    check_type(original, perturbed)
    cuda = is_cuda(original)
    assert is_cuda(perturbed) is cuda

    assert len(original.shape) == len(original.shape), (original.shape, perturbed.shape)
    for d in range(len(original.shape)):
        assert original.shape[d] == perturbed.shape[d], (original.shape, perturbed.shape, d)

    size = original.shape
    original = original.view(-1)
    perturbed = perturbed.view(-1)
    assert epsilon >= 0, epsilon

    difference = torch.abs(perturbed - original)
    sorted_difference, sorted_indices = torch.sort(difference, descending=True)
    sorted_original = torch.gather(original, dim=0, index=sorted_indices)
    # will hold the projection later
    sorted_projected = torch.gather(perturbed, dim=0, index=sorted_indices)

    # print(original)
    # print(perturbed)
    # print(difference)
    # print(sorted_original)
    # print(sorted_projected)
    # print(sorted_difference)

    k = int(math.ceil(epsilon))
    k = min(k, sorted_difference.size(0) - 1)
    assert k > 0, k

    projection_sorted_original = sorted_original[:k]
    projection_sorted_perturbed = sorted_projected[:k]
    projection_sorted_projected = int_msb_projection(projection_sorted_original, projection_sorted_perturbed)

    sorted_projected[:k] = projection_sorted_projected
    sorted_projected[k:] = sorted_original[k:]
    # sorted_projected[k:] = 0

    # print('---')
    # print(sorted_projected)

    # projected = torch.zeros_like(sorted_projected).scatter_(dim=0, index=sorted_indices, src=sorted_projected)
    projected = torch.scatter(sorted_projected, dim=0, index=sorted_indices, src=sorted_projected)
    # print(projected)

    return projected.view(size)


def int_hamming_distance(a, b):
    """
    Bit-wise and between float tensors.

    :param a: first tensor
    :type a: torch.Tensor
    :param b: first tensor
    :type b: torch.Tensor
    :return: bit-wise and
    :rtype: torch.Tensor
    """

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    check_type(a, b)
    cuda = is_cuda(a)
    assert is_cuda(b) is cuda

    assert len(a.shape) == len(b.shape), (a.shape, b.shape)
    for d in range(len(a.shape)):
        assert a.shape[d] == b.shape[d], (a.shape, b.shape, d)

    if cuda:
        dist = torch.cuda.IntTensor(a.shape).fill_(0)
    else:
        dist = torch.IntTensor(a.shape).fill_(0)

    n = a.nelement()
    shape = list(dist.shape)
    grid, block = cupy.grid_block(shape)

    type = str(a.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%shammingdistance' % type)(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  a.data_ptr(),
                  b.data_ptr(),
                  dist.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _dist = cffi.ffi.cast('int*', dist.data_ptr())

        if type == 'int32':
            _a = cffi.ffi.cast('int*', a.data_ptr())
            _b = cffi.ffi.cast('int*', b.data_ptr())
            cffi.lib.cffi_int32hammingdistance(_n, _a, _b, _dist)
        elif type == 'int16':
            _a = cffi.ffi.cast('short*', a.data_ptr())
            _b = cffi.ffi.cast('short*', b.data_ptr())
            cffi.lib.cffi_int16hammingdistance(_n, _a, _b, _dist)
        elif type == 'int8':
            _a = cffi.ffi.cast('char*', a.data_ptr())
            _b = cffi.ffi.cast('char*', b.data_ptr())
            cffi.lib.cffi_int8hammingdistance(_n, _a, _b, _dist)
        elif type == 'uint8':
            _a = cffi.ffi.cast('unsigned char*', a.data_ptr())
            _b = cffi.ffi.cast('unsigned char*', b.data_ptr())
            cffi.lib.cffi_uint8hammingdistance(_n, _a, _b, _dist)
        else:
            raise NotImplementedError

    return dist


# 0.5^9
INT32_FAST_RANDOM_FLIP_0001953125 = '&&&&&&&&'
# 0.5^8 * 0.75
INT32_FAST_RANDOM_FLIP_0002929688 = '|&&&&&&&&'
# 0.5^8
INT32_FAST_RANDOM_FLIP_000390625 = '&&&&&&&'
# 0.5^7 * 0.75
INT32_FAST_RANDOM_FLIP_0005859375 = '|&&&&&&&'
# 0.5^7
INT32_FAST_RANDOM_FLIP_00078125 = '&&&&&&'
# 0.5^6 * 0.75
INT32_FAST_RANDOM_FLIP_001171875 = '|&&&&&&'
# 0.5^6
INT32_FAST_RANDOM_FLIP_0015625 = '&&&&&'
# 0.5^5 * 0.75
INT32_FAST_RANDOM_FLIP_00234375 = '|&&&&&'
# 0.5^5
INT32_FAST_RANDOM_FLIP_003125 = '&&&&'


def int_fast_random_flip(input, prob=INT32_FAST_RANDOM_FLIP_001171875, protected_bits=[0]*32):
    """
    Fast version of random int32 bit flips supporting only specific flip probabilities.

    Inspired by https://stackoverflow.com/questions/35795110/fast-way-to-generate-pseudo-random-bits-with-a-given-probability-of-0-or-1-for-e/35811904#35811904.

    Protected bits will be ensured by converting protected_bits to int, and then anding the mask with it before applying xor for bit flips.

    Important: underestimates probabilities slightly!

    :param input: input tensor
    :type input: torch.Tensor
    :param prob: probability of a flip per bit
    :type prob: float
    :param protected_bits:
    :param protected_bits: list of length 32, indicating whether a bit can be flipped (1) or not (0)
    :type protected_bits: [int]
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    # assert (input.is_contiguous() == True)
    if not input.is_contiguous():
        input = input.contiguous()
    assert (input.dtype == torch.int32)

    assert prob in [
        INT32_FAST_RANDOM_FLIP_0001953125,
        INT32_FAST_RANDOM_FLIP_0002929688,
        INT32_FAST_RANDOM_FLIP_000390625,
        INT32_FAST_RANDOM_FLIP_0005859375,
        INT32_FAST_RANDOM_FLIP_00078125,
        INT32_FAST_RANDOM_FLIP_001171875,
        INT32_FAST_RANDOM_FLIP_0015625,
        INT32_FAST_RANDOM_FLIP_00234375,
        INT32_FAST_RANDOM_FLIP_003125,
    ]

    def generator(pattern, size, cuda=False):
        if cuda:
            r = torch.cuda.IntTensor(*size).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
        else:
            r = torch.IntTensor(*size).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)

        for i in range(len(pattern)):
            if cuda:
                a = torch.cuda.IntTensor(*size).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
            else:
                a = torch.IntTensor(*size).random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)

            if pattern[i] == '&':
                r = int_and(r, a)
            elif pattern[i] == '|':
                r = int_or(r, a)
        return r

    bits = ''
    for protected_bit in protected_bits:
        if protected_bit == 1:
            bits += '0'
        else:
            bits += '1'
    protected = int(bits, 2)

    size = list(input.shape)
    protected = torch.ones(size, dtype=torch.int32)*protected
    if is_cuda(input):
        protected = protected.cuda()

    random = generator(prob, size, cuda=is_cuda(input))
    random = int_and(random, protected)
    output = int_xor(input, random)
    return output


def int_flip(input, mask, precision=None):
    """
    Flip bits in input according to mask

    :param input: input tensor
    :type input: torch.Tensor
    :param mask: boolean mask
    :type: mask: torch.Tensor
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)

    assert (mask.dtype == torch.bool)
    assert is_cuda(mask) is cuda

    assert len(input.shape) + 1 == len(mask.shape), (input.shape, mask.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == mask.shape[d], (input.shape, mask.shape, d)
    assert mask.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%sflip' % type)(
            # https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  mask.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _mask = cffi.ffi.cast('bool*', mask.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32flip(_n, _mask, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16flip(_n, _mask, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8flip(_n, _mask, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8flip(_n, _mask, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_set_zero(input, m, precision=None):
    """
    Set the m LSBs to zero.

    :param input: input tensor
    :type input: torch.Tensor
    :param m: number of LSBs
    :type m: int
    :return: input with m LBSs set to zero
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)
    assert m <= precision

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%ssetzero' % type)(
            # https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  m,
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _m = cffi.ffi.cast('int*', m)

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32setzero(_n, _m, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16setzero(_n, _m, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8setzero(_n, _m, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8setzero(_n, _m, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_set(input, set1, set0, precision=None):
    """
    Flip bits in input according to mask

    :param input: input tensor
    :type input: torch.Tensor
    :param set1: boolean mask to set to 1
    :type: set1: torch.Tensor
    :param set0: boolean mask to set to 0
    :type set0: torch.Tensor
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not set1.is_contiguous():
        set1 = set1.contiguous()
    if not set0.is_contiguous():
        set0 = set0.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)

    assert (set1.dtype == torch.bool)
    assert is_cuda(set1) is cuda
    assert (set0.dtype == torch.bool)
    assert is_cuda(set0) is cuda

    assert len(input.shape) + 1 == len(set1.shape), (input.shape, set1.shape)
    assert len(input.shape) + 1 == len(set0.shape), (input.shape, set0.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == set1.shape[d], (input.shape, set1.shape, d)
        assert input.shape[d] == set0.shape[d], (input.shape, set0.shape, d)
    assert set1.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert set0.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%sset' % type)(
            # https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  set1.data_ptr(),
                  set0.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _set1 = cffi.ffi.cast('bool*', set1.data_ptr())
        _set0 = cffi.ffi.cast('bool*', set0.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32set(_n, _set1, _set0, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16set(_n, _set1, _set0, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8set(_n, _set1, _set0, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8set(_n, _set1, _set0, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_random_flip(input, zero_prob=0.1, one_prob=0.1, protected_bits=[0]*32, rand=None, precision=None):
    """
    Randomly flip bits in a int32 tensor with the given probability to flip zeros or ones.

    Note that for zero and one probability of 0.1, the actually changed values are roughly a fraction of 0.075;
    in contrast to 0.092 for the cupy version.

    :param input: input tensor
    :type input: torch.Tensor
    :param rand: optional tensor holding random value per bit, shape is input.shape + [32]
    :type: rand: torch.Tensor
    :param zero_prob: probability to flip a zero
    :type zero_prob: float
    :param one_prob: probability to flip a one
    :type one_prob: float
    :param protected_bits: list of length 32, indicating whether a bit can be flipped (1) or not (0)
    :type protected_bits: [int]
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)
    if rand is None:
        rand = torch.rand(list(input.shape) + [precision])
        if cuda:
            rand = rand.cuda()
    if not rand.is_contiguous():
        rand = rand.contiguous()

    assert (rand.dtype == torch.float)
    assert is_cuda(rand) is cuda

    assert len(input.shape) + 1 == len(rand.shape), (input.shape, rand.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == rand.shape[d], (input.shape, rand.shape, d)
    assert rand.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert len(protected_bits) == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    zero_prob = torch.tensor(zero_prob, dtype=torch.float)
    one_prob = torch.tensor(one_prob, dtype=torch.float)
    protected_bits = torch.tensor(protected_bits, dtype=torch.int32)

    if cuda:
        zero_prob = zero_prob.cuda()
        one_prob = one_prob.cuda()
        protected_bits = protected_bits.cuda()

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)

    len_protected_bits = torch.tensor(len(protected_bits), dtype=torch.int32)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%srandomflip' % type)(
            # https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  zero_prob.data_ptr(),
                  one_prob.data_ptr(),
                  protected_bits.data_ptr(),
                  len_protected_bits.data_ptr(),
                  rand.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _zero_prob = cffi.ffi.cast('float*', zero_prob.data_ptr())
        _one_prob = cffi.ffi.cast('float*', one_prob.data_ptr())
        _protected_bits = cffi.ffi.cast('int*', protected_bits.data_ptr())
        _len_protected_bits = cffi.ffi.cast('int*', len_protected_bits.data_ptr())
        _rand = cffi.ffi.cast('float*', rand.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32randomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16randomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8randomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8randomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_masked_random_flip(input, mask, zero_prob=0.1, one_prob=0.1, protected_bits=[0]*32, rand=None, precision=None):
    """
    Randomly flip bits in a int32 tensor with the given probability to flip zeros or ones.

    The mask decides which values are subject to bit flips and which are not.

    Note that for zero and one probability of 0.1, the actually changed values are roughly a fraction of 0.075;
    in contrast to 0.092 for the cupy version.

    :param input: input tensor
    :type input: torch.Tensor
    :param mask: mask tensor, determining which values can be changed
    :type mask: torch.Tensor
    :param rand: optional tensor holding random value per bit, shape is input.shape + [32]
    :type: rand: torch.Tensor
    :param zero_prob: probability to flip a zero
    :type zero_prob: float
    :param one_prob: probability to flip a one
    :type one_prob: float
    :param protected_bits: list of length 32, indicating whether a bit can be flipped (1) or not (0)
    :type protected_bits: [int]
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)
    if rand is None:
        rand = torch.rand(list(input.shape) + [precision])
        if cuda:
            rand = rand.cuda()
    if not rand.is_contiguous():
        rand = rand.contiguous()

    assert (rand.dtype == torch.float)
    assert is_cuda(rand) is cuda

    assert (mask.dtype == torch.bool)
    assert is_cuda(mask) is cuda

    assert len(input.shape) == len(mask.shape), (input.shape, mask.shape)
    assert len(input.shape) + 1 == len(rand.shape), (input.shape, rand.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == rand.shape[d], (input.shape, rand.shape, d)
        assert input.shape[d] == mask.shape[d], (input.shape, mask.shape, d)
    assert rand.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert len(protected_bits) == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    zero_prob = torch.tensor(zero_prob, dtype=torch.float)
    one_prob = torch.tensor(one_prob, dtype=torch.float)
    protected_bits = torch.tensor(protected_bits, dtype=torch.int32)

    if cuda:
        zero_prob = zero_prob.cuda()
        one_prob = one_prob.cuda()
        protected_bits = protected_bits.cuda()

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)
    len_protected_bits = torch.tensor(len(protected_bits), dtype=torch.int32)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%smaskedrandomflip' % type)(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  zero_prob.data_ptr(),
                  one_prob.data_ptr(),
                  protected_bits.data_ptr(),
                  len_protected_bits.data_ptr(),
                  mask.data_ptr(),
                  rand.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _zero_prob = cffi.ffi.cast('float*', zero_prob.data_ptr())
        _one_prob = cffi.ffi.cast('float*', one_prob.data_ptr())
        _protected_bits = cffi.ffi.cast('int*', protected_bits.data_ptr())
        _len_protected_bits = cffi.ffi.cast('int*', len_protected_bits.data_ptr())
        _mask = cffi.ffi.cast('bool*', mask.data_ptr())
        _rand = cffi.ffi.cast('float*', rand.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32maskedrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16maskedrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8maskedrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8maskedrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_individual_random_flip(input, zero_prob, one_prob, protected_bits=[0]*32, rand=None, precision=None):
    """
    Randomly flip bits in a int32 tensor with the given probability to flip zeros or ones.

    Note that for zero and one probability of 0.1, the actually changed values are roughly a fraction of 0.075;
    in contrast to 0.092 for the cupy version.

    :param input: input tensor
    :type input: torch.Tensor
    :param rand: optional tensor holding random value per bit, shape is input.shape + [32]
    :type: rand: torch.Tensor
    :param zero_prob: tensor for per-bit probabilities to flip a zero
    :type zero_prob: torch.Tensor
    :param one_prob: tensor of per-bit probabilities to flip a one
    :type one_prob: torch.Tensor
    :param protected_bits: list of length 32, indicating whether a bit can be flipped (1) or not (0)
    :type protected_bits: [int]
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not zero_prob.is_contiguous():
        zero_prob = zero_prob.contiguous()
    if not one_prob.is_contiguous():
        one_prob = one_prob.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)
    if rand is None:
        rand = torch.rand(list(input.shape) + [precision])
        if cuda:
            rand = rand.cuda()
    if not rand.is_contiguous():
        rand = rand.contiguous()

    assert (rand.dtype == torch.float)
    assert is_cuda(rand) is cuda

    assert (zero_prob.dtype == torch.float)
    assert is_cuda(zero_prob) is cuda

    assert (one_prob.dtype == torch.float)
    assert is_cuda(one_prob) is cuda

    assert len(input.shape) + 1 == len(rand.shape), (input.shape, rand.shape)
    assert len(input.shape) + 1 == len(zero_prob.shape), (input.shape, zero_prob.shape)
    assert len(input.shape) + 1 == len(one_prob.shape), (input.shape, one_prob.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == rand.shape[d], (input.shape, rand.shape, d)
        assert input.shape[d] == zero_prob.shape[d], (input.shape, zero_prob.shape, d)
        assert input.shape[d] == one_prob.shape[d], (input.shape, one_prob.shape, d)
    assert rand.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert zero_prob.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert one_prob.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert len(protected_bits) == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    protected_bits = torch.tensor(protected_bits, dtype=torch.int32)
    if cuda:
        protected_bits = protected_bits.cuda()

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)
    len_protected_bits = torch.tensor(len(protected_bits), dtype=torch.int32)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%sindividualrandomflip' % type)(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  zero_prob.data_ptr(),
                  one_prob.data_ptr(),
                  protected_bits.data_ptr(),
                  len_protected_bits.data_ptr(),
                  rand.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _zero_prob = cffi.ffi.cast('float*', zero_prob.data_ptr())
        _one_prob = cffi.ffi.cast('float*', one_prob.data_ptr())
        _protected_bits = cffi.ffi.cast('int*', protected_bits.data_ptr())
        _len_protected_bits = cffi.ffi.cast('int*', len_protected_bits.data_ptr())
        _rand = cffi.ffi.cast('float*', rand.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32individualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16individualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8individualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8individualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _rand, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_masked_individual_random_flip(input, mask, zero_prob, one_prob, protected_bits=[0]*32, rand=None, precision=None):
    """
    Randomly flip bits in a int32 tensor with the given probability to flip zeros or ones.

    The mask decides which values are subject to bit flips and which are not.

    Note that for zero and one probability of 0.1, the actually changed values are roughly a fraction of 0.075;
    in contrast to 0.092 for the cupy version.

    :param input: input tensor
    :type input: torch.Tensor
    :param mask: mask tensor, determining which values can be changed
    :type mask: torch.Tensor
    :param rand: optional tensor holding random value per bit, shape is input.shape + [32]
    :type: rand: torch.Tensor
    :param zero_prob: tensor for per-bit probabilities to flip a zero
    :type zero_prob: torch.Tensor
    :param one_prob: tensor of per-bit probabilities to flip a one
    :type one_prob: torch.Tensor
    :param protected_bits: list of length 32, indicating whether a bit can be flipped (1) or not (0)
    :type protected_bits: [int]
    :return: input with random bit flips
    :rtype: torch.Tensor
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not zero_prob.is_contiguous():
        zero_prob = zero_prob.contiguous()
    if not one_prob.is_contiguous():
        one_prob = one_prob.contiguous()

    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision
    cuda = is_cuda(input)
    if rand is None:
        rand = torch.rand(list(input.shape) + [precision])
        if cuda:
            rand = rand.cuda()
    if not rand.is_contiguous():
        rand = rand.contiguous()

    assert (rand.dtype == torch.float)
    assert is_cuda(rand) is cuda

    assert (mask.dtype == torch.bool)
    assert is_cuda(mask) is cuda

    assert (zero_prob.dtype == torch.float)
    assert is_cuda(zero_prob) is cuda

    assert (one_prob.dtype == torch.float)
    assert is_cuda(one_prob) is cuda

    assert len(input.shape) == len(mask.shape), (input.shape, mask.shape)
    assert len(input.shape) + 1 == len(rand.shape), (input.shape, rand.shape)
    assert len(input.shape) + 1 == len(zero_prob.shape), (input.shape, zero_prob.shape)
    assert len(input.shape) + 1 == len(one_prob.shape), (input.shape, one_prob.shape)
    for d in range(len(input.shape)):
        assert input.shape[d] == rand.shape[d], (input.shape, rand.shape, d)
        assert input.shape[d] == mask.shape[d], (input.shape, mask.shape, d)
        assert input.shape[d] == zero_prob.shape[d], (input.shape, zero_prob.shape, d)
        assert input.shape[d] == one_prob.shape[d], (input.shape, one_prob.shape, d)
    assert rand.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert zero_prob.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert one_prob.shape[-1] == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)
    assert len(protected_bits) == precision, 'precision does not match, using inferred precision: %s' % (inferred_precision == precision)

    protected_bits = torch.tensor(protected_bits, dtype=torch.int32)
    if cuda:
        protected_bits = protected_bits.cuda()

    output = input.new_zeros(input.shape)
    n = output.nelement()
    shape = list(output.shape)
    grid, block = cupy.grid_block(shape)
    len_protected_bits = torch.tensor(len(protected_bits), dtype=torch.int32)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%smaskedindividualrandomflip' % type)(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  zero_prob.data_ptr(),
                  one_prob.data_ptr(),
                  protected_bits.data_ptr(),
                  len_protected_bits.data_ptr(),
                  mask.data_ptr(),
                  rand.data_ptr(),
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _zero_prob = cffi.ffi.cast('float*', zero_prob.data_ptr())
        _one_prob = cffi.ffi.cast('float*', one_prob.data_ptr())
        _protected_bits = cffi.ffi.cast('int*', protected_bits.data_ptr())
        _len_protected_bits = cffi.ffi.cast('int*', len_protected_bits.data_ptr())
        _mask = cffi.ffi.cast('bool*', mask.data_ptr())
        _rand = cffi.ffi.cast('float*', rand.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            _output = cffi.ffi.cast('int*', output.data_ptr())
            cffi.lib.cffi_int32maskedindividualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            _output = cffi.ffi.cast('short*', output.data_ptr())
            cffi.lib.cffi_int16maskedindividualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            _output = cffi.ffi.cast('char*', output.data_ptr())
            cffi.lib.cffi_int8maskedindividualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            _output = cffi.ffi.cast('unsigned char*', output.data_ptr())
            cffi.lib.cffi_uint8maskedindividualrandomflip(_n, _zero_prob, _one_prob, _protected_bits, _len_protected_bits, _mask, _rand, _input, _output)
        else:
            raise NotImplementedError

    return output


def int_bits(input, precision=None):
    """
    Read individual bits in bool tensor.

    :param input: input tensor
    :type input: torch.Tensor
    :return: bit tensor
    :rtype: torch.Tensor
    """

    #assert (input.is_contiguous() == True)
    if not input.is_contiguous():
        input = input.contiguous()
    inferred_precision = check_type(input)
    if precision is None:
        precision = inferred_precision

    cuda = is_cuda(input)
    output = torch.zeros(list(input.shape) + [precision], dtype=torch.bool)
    if cuda:
        output = output.cuda()
    n = input.nelement()
    shape = list(input.shape)
    grid, block = cupy.grid_block(shape)

    type = str(input.dtype).replace('torch.', '')
    if cuda:
        cupy.cunnex('cupy_%sbits' % type)(
            grid=tuple(grid),
            block=tuple(block),
            args=[n,
                  input.data_ptr(),
                  output.data_ptr()],
            stream=cupy.Stream
        )
    else:
        _n = cffi.ffi.cast('int', n)
        _output = cffi.ffi.cast('bool*', output.data_ptr())

        if type == 'int32':
            _input = cffi.ffi.cast('int*', input.data_ptr())
            cffi.lib.cffi_int32bits(_n, _input, _output)
        elif type == 'int16':
            _input = cffi.ffi.cast('short*', input.data_ptr())
            cffi.lib.cffi_int16bits(_n, _input, _output)
        elif type == 'int8':
            _input = cffi.ffi.cast('char*', input.data_ptr())
            cffi.lib.cffi_int8bits(_n, _input, _output)
        elif type == 'uint8':
            _input = cffi.ffi.cast('unsigned char*', input.data_ptr())
            cffi.lib.cffi_uint8bits(_n, _input, _output)
        else:
            raise NotImplementedError

    return output