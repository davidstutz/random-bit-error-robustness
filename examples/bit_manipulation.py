import sys
import os
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import torch
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
