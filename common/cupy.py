"""
Redundancy is motiviated by reducing memory overhead for the simple versions (without masks and individual proabbilities).
"""
import numpy
import torch

try:
    import cupy


    @cupy.util.memoize(for_each_device=True)
    def cunnex(strFunction):
        assert strFunction in globals().keys()
        return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
except ImportError:
    print("CUPY cannot initialize, not using CUDA kernels")


class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


def grid_block(shape):
    """
    Get grid and block dimensions from tensor shape.

    :param shape: shape
    :type shape: (int9
    :return: grid, block
    :rtype: (int), (int)
    """

    grid = (numpy.prod(shape), 1, 1)

    # allowed number of threads per block: 65,535
    N_XY_BLOCKS = 512 # 1024
    N_Z_BLOCKS = 32 # 64

    # allowed dimensions: https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
    actual_grid = [int((grid[0] + N_XY_BLOCKS - 1) / N_XY_BLOCKS), int((grid[1] + N_XY_BLOCKS - 1) / N_XY_BLOCKS), int((grid[2] + N_Z_BLOCKS - 1) / N_Z_BLOCKS)]
    actual_block = [min(N_XY_BLOCKS, grid[0]), min(N_XY_BLOCKS, grid[1]), min(N_Z_BLOCKS, grid[2])]

    return actual_grid, actual_block


cupy_int32and = '''
    extern "C" __global__ void cupy_int32and(
        const int n,
        const int* a,
        const int* b,
        int* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
'''

cupy_int32or = '''
    extern "C" __global__ void cupy_int32or(
        const int n,
        const int* a,
        const int* b,
        int* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
'''

cupy_int32xor = '''
    extern "C" __global__ void cupy_int32xor(
        const int n,
        const int* a,
        const int* b,
        int* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
'''

cupy_int32msbprojection= '''
    extern "C" __global__ void cupy_int32msbprojection(
        const int n,
        const int* original,
        const int* perturbed,
        int* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (elem_idx >= n) {
            return;
        }
    
        output[elem_idx] = original[elem_idx];

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
            int mask =  1 << bit_idx;

            int original_bit = original[elem_idx] & mask;
            int perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }

'''

cupy_int32hammingdistance = '''
    extern "C" __global__ void cupy_int32hammingdistance(
        const int n,
        const int* a,
        const int* b,
        int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        int x = a[elem_idx] ^ b[elem_idx];
         
        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
'''

cupy_int32flip = '''
    extern "C" __global__ void cupy_int32flip(
        const int n,
        const bool* mask,
        const int* input,
        int* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        int xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if(mask[32*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int32set = '''
    extern "C" __global__ void cupy_int32set(
        const int n,
        const bool* set1,
        const bool* set0,
        const int* input,
        int* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        int set1_mask = 0;
        int set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (set1[32*elem_idx + bit_idx]) {
                set1_mask |= (1 << bit_idx);
            }
            if (set0[32*elem_idx + bit_idx]) {
                set0_mask |= (1 << bit_idx);
            }
        }

        int output_elem = input[elem_idx];
        output_elem |= set1_mask;
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting

        output[elem_idx] = output_elem;
    }
'''

cupy_int32setzero = '''
    extern "C" __global__ void cupy_int32setzero(
        const int n,
        const int m,
        const int* input,
        int* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        int set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            if (bit_idx < m) {
                // first bit is lSB if little endian (Linux 32- or 64-bit)
                set0_mask |= (1 << bit_idx);
            }
        }

        int output_elem = input[elem_idx];
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting
        output[elem_idx] = output_elem;
    }
'''

cupy_int32randomflip = '''
    extern "C" __global__ void cupy_int32randomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const int* input,
        int* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        int xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if(!protected_bits[bit_idx]) {
                int input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if(rand_src[32*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int32maskedrandomflip = '''
    extern "C" __global__ void cupy_int32maskedrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const int* input,
        int* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        int xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if(!protected_bits[bit_idx]) {
                int input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if(rand_src[32*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int32individualrandomflip = '''
    extern "C" __global__ void cupy_int32individualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const int* input,
        int* output 
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        int xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if(!protected_bits[bit_idx]) {
                int input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;

                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[32*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[32*elem_idx + bit_idx];
                }

                if(rand_src[32*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int32maskedindividualrandomflip = '''
    extern "C" __global__ void cupy_int32maskedindividualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const int* input,
        int* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int32maskedindividualrandomflip");
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        int xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if(!protected_bits[bit_idx]) {
                int input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[32*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[32*elem_idx + bit_idx];
                }

                if(rand_src[32*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int32bits = '''
    extern "C" __global__ void cupy_int32bits(
        const int n,
        const int* input,
        bool* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int32bits");
            return;
        }

        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            int mask =  1 << bit_idx;
            int masked_input = input[elem_idx] & mask;
            int bit = masked_input >> bit_idx;
            
            output[32*elem_idx + bit_idx] = bit;
        }
    }
'''

cupy_int16and = '''
    extern "C" __global__ void cupy_int16and(
        const int n,
        const short* a,
        const short* b,
        short* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int16and");
            return;
        }

        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
'''

cupy_int16or = '''
    extern "C" __global__ void cupy_int16or(
        const int n,
        const short* a,
        const short* b,
        short* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int16or");
            return;
        }

        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
'''

cupy_int16xor = '''
    extern "C" __global__ void cupy_int16xor(
        const int n,
        const short* a,
        const short* b,
        short* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int16xor");
            return;
        }

        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
'''

cupy_int16msbprojection = '''
    extern "C" __global__ void cupy_int16msbprojection(
        const int n,
        const short* original,
        const short* perturbed,
        short* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) { 
            return; 
        } 
 
        output[elem_idx] = original[elem_idx]; 

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 15; bit_idx >= 0; bit_idx--) {
            short mask = 1 << bit_idx;
            short original_bit = original[elem_idx] & mask;
            short perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }

'''

cupy_int16hammingdistance = '''
    extern "C" __global__ void cupy_int16hammingdistance(
        const int n,
        const short* a,
        const short* b,
        int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        dist[elem_idx] = 0;
        short x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
'''

cupy_int16flip = '''
    extern "C" __global__ void cupy_int16flip(
        const int n,
        const bool* mask,
        const short* input,
        short* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        short xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (mask[16*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int16set = '''
    extern "C" __global__ void cupy_int16set(
        const int n,
        const bool* set1,
        const bool* set0,
        const short* input,
        short* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        short set1_mask = 0;
        short set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (set1[16*elem_idx + bit_idx]) {
                set1_mask |= (1 << bit_idx);
            }
            if (set0[16*elem_idx + bit_idx]) {
                set0_mask |= (1 << bit_idx);
            }
        }

        short output_elem = input[elem_idx];
        output_elem |= set1_mask;
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting

        output[elem_idx] = output_elem;
    }
'''

cupy_int16setzero = '''
    extern "C" __global__ void cupy_int16setzero(
        const int n,
        const int m,
        const short* input,
        short* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        short set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            if (bit_idx < m) {
                // first bit is lSB if little endian (Linux 32- or 64-bit)
                set0_mask |= (1 << bit_idx);
            }
        }
 
        short output_elem = input[elem_idx];
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting
        output[elem_idx] = output_elem;
    }
'''

cupy_int16randomflip = '''
    extern "C" __global__ void cupy_int16randomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const short* input,
        short* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        short xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                short input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[16*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int16maskedrandomflip = '''
    extern "C" __global__ void cupy_int16maskedrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const short* input,
        short* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        short xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                short input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[16*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int16individualrandomflip = '''
    extern "C" __global__ void cupy_int16individualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const short* input,
        short* output 
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        short xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                short input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;

                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[16*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[16*elem_idx + bit_idx];
                }

                if (rand_src[16*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int16maskedindividualrandomflip = '''
    extern "C" __global__ void cupy_int16maskedindividualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const short* input,
        short* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        short xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                short input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[16*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[16*elem_idx + bit_idx];
                }

                if (rand_src[16*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int16bits = '''
    extern "C" __global__ void cupy_int16bits(
        const int n,
        const short* input,
        bool* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            short mask =  1 << bit_idx;
            short masked_input = input[elem_idx] & mask;
            short bit = masked_input >> bit_idx;

            output[16*elem_idx + bit_idx] = bit;
        }
    }
'''

cupy_int8and = '''
    extern "C" __global__ void cupy_int8and(
        const int n,
        const char* a,
        const char* b,
        char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
'''

cupy_int8or = '''
    extern "C" __global__ void cupy_int8or(
        const int n,
        const char* a,
        const char* b,
        char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
'''

cupy_int8xor = '''
    extern "C" __global__ void cupy_int8xor(
        const int n,
        const char* a,
        const char* b,
        char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
'''

cupy_int8msbprojection = '''
    extern "C" __global__ void cupy_int8msbprojection(
        const int n,
        const char* original,
        const char* perturbed,
        char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) { 
            return; 
        } 

        output[elem_idx] = original[elem_idx]; 

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            char mask = 1 << bit_idx;
            char original_bit = original[elem_idx] & mask;
            char perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }

'''

cupy_int8hammingdistance = '''
    extern "C" __global__ void cupy_int8hammingdistance(
        const int n,
        const char* a,
        const char* b,
        int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        dist[elem_idx] = 0;
        char x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
'''

cupy_int8flip = '''
    extern "C" __global__ void cupy_int8flip(
        const int n,
        const bool* mask,
        const char* input,
        char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (mask[8*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int8set = '''
    extern "C" __global__ void cupy_int8set(
        const int n,
        const bool* set1,
        const bool* set0,
        const char* input,
        char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        char set1_mask = 0;
        char set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (set1[8*elem_idx + bit_idx]) {
                set1_mask |= (1 << bit_idx);
            }
            if (set0[8*elem_idx + bit_idx]) {
                set0_mask |= (1 << bit_idx);
            }
        }

        char output_elem = input[elem_idx];
        output_elem |= set1_mask;
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting

        output[elem_idx] = output_elem;
    }
'''

cupy_int8setzero = '''
    extern "C" __global__ void cupy_int8setzero(
        const int n,
        const int m,
        const char* input,
        char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        char set0_mask = 0; 
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            if (bit_idx < m) {
                // first bit is lSB if little endian (Linux 32- or 64-bit)
                set0_mask |= (1 << bit_idx);
            }
        }

        char output_elem = input[elem_idx];
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting
        output[elem_idx] = output_elem;
    }
'''

cupy_int8randomflip = '''
    extern "C" __global__ void cupy_int8randomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const char* input,
        char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int8maskedrandomflip = '''
    extern "C" __global__ void cupy_int8maskedrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const char* input,
        char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int8individualrandomflip = '''
    extern "C" __global__ void cupy_int8individualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const char* input,
        char* output 
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        char xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;

                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[8*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[8*elem_idx + bit_idx];
                }

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_int8maskedindividualrandomflip = '''
    extern "C" __global__ void cupy_int8maskedindividualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const char* input,
        char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        char xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[8*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[8*elem_idx + bit_idx];
                }

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                } 
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    } 
'''

cupy_int8bits = '''
    extern "C" __global__ void cupy_int8bits(
        const int n,
        const char* input,
        bool* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            char mask =  1 << bit_idx;
            char masked_input = input[elem_idx] & mask;
            char bit = masked_input >> bit_idx;

            output[8*elem_idx + bit_idx] = bit;
        }
    }
'''

cupy_uint8and = '''
    extern "C" __global__ void cupy_uint8and(
        const int n,
        const unsigned char* a,
        const unsigned char* b,
        unsigned char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
'''

cupy_uint8or = '''
    extern "C" __global__ void cupy_uint8or(
        const int n,
        const unsigned char* a,
        const unsigned char* b,
        unsigned char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
'''

cupy_uint8xor = '''
    extern "C" __global__ void cupy_uint8xor(
        const int n,
        const unsigned char* a,
        const unsigned char* b,
        unsigned char* c
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
'''

cupy_uint8msbprojection = '''
    extern "C" __global__ void cupy_uint8msbprojection(
        const int n,
        const unsigned char* original,
        const unsigned char* perturbed,
        unsigned char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) { 
            return; 
        } 

        output[elem_idx] = original[elem_idx]; 

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            unsigned char mask = 1 << bit_idx;
            unsigned char original_bit = original[elem_idx] & mask;
            unsigned char perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }

'''

cupy_uint8hammingdistance = '''
    extern "C" __global__ void cupy_uint8hammingdistance(
        const int n,
        const unsigned char* a,
        const unsigned char* b,
        int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        dist[elem_idx] = 0;
        unsigned char x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
'''

cupy_uint8flip = '''
    extern "C" __global__ void cupy_uint8flip(
        const int n,
        const bool* mask,
        const unsigned char* input,
        unsigned char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        unsigned char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (mask[8*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_uint8set = '''
    extern "C" __global__ void cupy_uint8set(
        const int n,
        const bool* set1,
        const bool* set0,
        const unsigned char* input,
        unsigned char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        unsigned char set1_mask = 0;
        unsigned char set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if (set1[8*elem_idx + bit_idx]) {
                set1_mask |= (1 << bit_idx);
            }
            if (set0[8*elem_idx + bit_idx]) {
                set0_mask |= (1 << bit_idx);
            }
        }

        unsigned char output_elem = input[elem_idx];
        output_elem |= set1_mask;
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting

        output[elem_idx] = output_elem;
    }
'''

cupy_uint8setzero = '''
    extern "C" __global__ void cupy_uint8setzero(
        const int n,
        const int m,
        const unsigned char* input,
        unsigned char* output
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        unsigned char set0_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            if (bit_idx < m) {
                // first bit is lSB if little endian (Linux 32- or 64-bit)
                set0_mask |= (1 << bit_idx);
            }
        }

        unsigned char output_elem = input[elem_idx];
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting
        output[elem_idx] = output_elem;
    }
'''

cupy_uint8randomflip = '''
    extern "C" __global__ void cupy_uint8randomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const unsigned char* input,
        unsigned char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        unsigned char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                unsigned char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_uint8maskedrandomflip = '''
    extern "C" __global__ void cupy_uint8maskedrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const unsigned char* input,
        unsigned char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        unsigned char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                unsigned char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                float bit_flip_prob = (input_bit == 1) ? *one_bit_flip_prob : *zero_bit_flip_prob;

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_uint8individualrandomflip = '''
    extern "C" __global__ void cupy_uint8individualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const float* rand_src,
        const unsigned char* input,
        unsigned char* output 
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        unsigned char xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                unsigned char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;

                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[8*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[8*elem_idx + bit_idx];
                }

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_uint8maskedindividualrandomflip = '''
    extern "C" __global__ void cupy_uint8maskedindividualrandomflip(
        const int n,
        const float* zero_bit_flip_prob,
        const float* one_bit_flip_prob,
        const int* protected_bits,
        const int* len_protected_bits,
        const bool* mask,
        const float* rand_src,
        const unsigned char* input,
        unsigned char* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            return;
        }

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
            return;
        }

        unsigned char xor_mask = 0;
        float bit_flip_prob;

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)

            if (!protected_bits[bit_idx]) {
                unsigned char input_bit = (input[elem_idx] & (1 << bit_idx)) >> bit_idx;
                if (input_bit == 1) {
                    bit_flip_prob = one_bit_flip_prob[8*elem_idx + bit_idx];
                }
                else {
                    bit_flip_prob = zero_bit_flip_prob[8*elem_idx + bit_idx];
                }

                if (rand_src[8*elem_idx + bit_idx] < bit_flip_prob) {
                    xor_mask |= (1 << bit_idx);
                }
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
'''

cupy_uint8bits = '''
    extern "C" __global__ void cupy_uint8bits(
        const int n,
        const unsigned char* input,
        bool* output
    ) {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
        int elem_idx = threadId;

        if (elem_idx >= n) {
            //printf("invalid elem_idx in int8bits");
            return;
        } 

        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            unsigned char mask =  1 << bit_idx;
            unsigned char masked_input = input[elem_idx] & mask;
            unsigned char bit = masked_input >> bit_idx;

            output[8*elem_idx + bit_idx] = bit;
        }
    }
'''