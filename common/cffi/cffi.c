#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

// Check little or big endian
// Linux (32- or 64-bit) is probably little endian
bool endian(void)
{
    // https://stackoverflow.com/questions/12791864/c-program-to-check-little-vs-big-endian
    volatile uint32_t i=0x01234567;
    // return 0 for big endian, 1 for little endian.
    return (*((uint8_t*)(&i))) == 0x67;
}

void cffi_int32and(
    const int n,
    const int* a,
    const int* b,
    int* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
}

void cffi_int32or(
    const int n,
    const int* a,
    const int* b,
    int* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
}

void cffi_int32xor(
    const int n,
    const int* a,
    const int* b,
    int* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
}

void cffi_int32msbprojection(
    const int n,
    const int* original,
    const int* perturbed,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
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
}

void cffi_int32hammingdistance(
    const int n,
    const int* a,
    const int* b,
    int* dist
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        dist[elem_idx] = 0;
        int x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
}

void cffi_int32flip(
    const int n,
    const bool* mask,
    const int* input,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int32set(
    const int n,
    const bool* set1,
    const bool* set0,
    const int* input,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int32setzero(
    const int n,
    const int m,
    const int* input,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        int set0_mask = 0;
        for (int bit_idx = 0; bit_idx <32; bit_idx ++) {
            if (bit_idx < m) {
                // first bit is lSB if little endian (Linux 32- or 64-bit)
                set0_mask |= (1 << bit_idx);
            }
        }

        int output_elem = input[elem_idx];
        output_elem &= (~set0_mask); // negation will do all bits set to 0 to 0 and all other's to 1, so and will do the setting
        output[elem_idx] = output_elem;
    }
}

void cffi_int32randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const int* input,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int32maskedrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int32individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const int* input,
    int* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int32maskedindividualrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int32bits(
    const int n,
    const int* input,
    bool* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        for (int bit_idx = 0; bit_idx < 32; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            int mask =  1 << bit_idx;
            int masked_input = input[elem_idx] & mask;
            int bit = masked_input >> bit_idx;

            output[32*elem_idx + bit_idx] = bit;
        }
    }
}

void cffi_int16and(
    const int n,
    const short* a,
    const short* b,
    short* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
}

void cffi_int16or(
    const int n,
    const short* a,
    const short* b,
    short* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
}

void cffi_int16xor(
    const int n,
    const short* a,
    const short* b,
    short* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
}

void cffi_int16msbprojection(
    const int n,
    const short* original,
    const short* perturbed,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        output[elem_idx] = original[elem_idx];

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 15; bit_idx >= 0; bit_idx--) {
            short mask =  1 << bit_idx;
            short original_bit = original[elem_idx] & mask;
            short perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }
}

void cffi_int16hammingdistance(
    const int n,
    const short* a,
    const short* b,
    int* dist
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        dist[elem_idx] = 0;
        short x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
}

void cffi_int16flip(
    const int n,
    const bool* mask,
    const short* input,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        short xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if(mask[16*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
}

void cffi_int16set(
    const int n,
    const bool* set1,
    const bool* set0,
    const short* input,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int16setzero(
    const int n,
    const int m,
    const short* input,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int16randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const short* input,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int16maskedrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int16individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const short* input,
    short* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int16maskedindividualrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int16bits(
    const int n,
    const short* input,
    bool* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        for (int bit_idx = 0; bit_idx < 16; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            short mask =  1 << bit_idx;
            short masked_input = input[elem_idx] & mask;
            short bit = masked_input >> bit_idx;

            output[16*elem_idx + bit_idx] = bit;
        }
    }
}

void cffi_int8and(
    const int n,
    const char* a,
    const char* b,
    char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
}

void cffi_int8or(
    const int n,
    const char* a,
    const char* b,
    char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
}

void cffi_int8xor(
    const int n,
    const char* a,
    const char* b,
    char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
}

void cffi_int8msbprojection(
    const int n,
    const char* original,
    const char* perturbed,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        output[elem_idx] = original[elem_idx];

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            char mask =  1 << bit_idx;
            char original_bit = original[elem_idx] & mask;
            char perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }
}

void cffi_int8hammingdistance(
    const int n,
    const char* a,
    const char* b,
    int* dist
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        dist[elem_idx] = 0;
        char x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
}

void cffi_int8flip(
    const int n,
    const bool* mask,
    const char* input,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if(mask[8*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
}

void cffi_int8set(
    const int n,
    const bool* set1,
    const bool* set0,
    const char* input,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int8setzero(
    const int n,
    const int m,
    const char* input,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int8randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const char* input,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int8maskedrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int8individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const char* input,
    char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_int8maskedindividualrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_int8bits(
    const int n,
    const char* input,
    bool* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            char mask =  1 << bit_idx;
            char masked_input = input[elem_idx] & mask;
            char bit = masked_input >> bit_idx;

            output[8*elem_idx + bit_idx] = bit;
        }
    }
}

void cffi_uint8and(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] & b[elem_idx];
    }
}

void cffi_uint8or(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] | b[elem_idx];
    }
}

void cffi_uint8xor(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        c[elem_idx] = a[elem_idx] ^ b[elem_idx];
    }
}

void cffi_uint8msbprojection(
    const int n,
    const unsigned char* original,
    const unsigned char* perturbed,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        output[elem_idx] = original[elem_idx];

        // bit_idx = 0 would be LSB on little-endian
        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            unsigned char mask =  1 << bit_idx;
            unsigned char original_bit = original[elem_idx] & mask;
            unsigned char perturbed_bit = perturbed[elem_idx] & mask;

            if (original_bit != perturbed_bit) {
                output[elem_idx] ^= mask;
                break;
            }
        }
    }
}

void cffi_uint8hammingdistance(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    int* dist
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        dist[elem_idx] = 0;
        unsigned char x = a[elem_idx] ^ b[elem_idx];

        while(x != 0) {
            x = x & (x-1);
            dist[elem_idx]++;
        }
    }
}

void cffi_uint8flip(
    const int n,
    const bool* mask,
    const unsigned char* input,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        unsigned char xor_mask = 0;
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            if(mask[8*elem_idx + bit_idx]) {
                xor_mask |= (1 << bit_idx);
            }
        }

        output[elem_idx] = input[elem_idx];
        output[elem_idx] ^= xor_mask;
    }
}

void cffi_uint8set(
    const int n,
    const bool* set1,
    const bool* set0,
    const unsigned char* input,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_uint8setzero(
    const int n,
    const int m,
    const unsigned char* input,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_uint8randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const unsigned char* input,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_uint8maskedrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_uint8individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const unsigned char* input,
    unsigned char* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

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
}

void cffi_uint8maskedindividualrandomflip(
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
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {

        if (!mask[elem_idx]) {
            output[elem_idx] = input[elem_idx];
        }
        else {
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
    }
}

void cffi_uint8bits(
    const int n,
    const unsigned char* input,
    bool* output
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
        for (int bit_idx = 0; bit_idx < 8; bit_idx ++) {
            // first bit is lSB if little endian (Linux 32- or 64-bit)
            unsigned char mask =  1 << bit_idx;
            unsigned char masked_input = input[elem_idx] & mask;
            unsigned char bit = masked_input >> bit_idx;

            output[8*elem_idx + bit_idx] = bit;
        }
    }
}