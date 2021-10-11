/**
 * Definitions for bit manipulation methods.
 *
 * Note, code is redundant to avoid having to setup multiple tensors (possibly in CUDA)
 * for the simplest version without mask and without individual probabilities.
 */

/**
 * Bitwise and.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int32and(
    const int n,
    const int* a,
    const int* b,
    int* c
);

/**
 * Bitwise or.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int32or(
    const int n,
    const int* a,
    const int* b,
    int* c
);

/**
 * Bitwise xor.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int32xor(
    const int n,
    const int* a,
    const int* b,
    int* c
);

/**
 * MSB Projection.
 *
 * @param n number of elements
 * @param original original tensor
 * @param perturbed perturbed tensor
 * @param output putput tensor
 */
void cffi_int32msbprojection(
    const int n,
    const int* original,
    const int* perturbed,
    int* output
);

/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_int32hammingdistance(
    const int n,
    const int* a,
    const int* b,
    int* dist
);

/**
 * Flip bits according to mask.
 *
 * @param n number of elements in input
 * @param mask mask array
 * @param input input array
 * @param output output array.
 */
void cffi_int32flip(
    const int n,
    const bool* mask,
    const int* input,
    int* output
);

/**
 * Set bits according to masks.
 *
 * @param n number of elements in input
 * @param set1 mask for setting to 1
 * @param set0 mask for setting to 0
 * @param input input array
 * @param output output array
 */
void cffi_int32set(
    const int n,
    const bool* set1,
    const bool* set0,
    const int* input,
    int* output
);

/**
 * Set m LSBs to zero.
 *
 * @param n number of elements in input
 * @param m number of LSBs
 * @param input input array
 * @param output output array
 */
void cffi_int32setzero(
    const int n,
    const int m,
    const int* input,
    int* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int32randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const int* input,
    int* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int32individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const int* input,
    int* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Get bits.
 *
 * @param n number of elements
 * @param input input tensor
 * @param output output tensor
 */
void cffi_int32bits(
    const int n,
    const int* input,
    bool* output
);

/**
 * Bitwise and.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int16and(
    const int n,
    const short* a,
    const short* b,
    short* c
);

/**
 * Bitwise or.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int16or(
    const int n,
    const short* a,
    const short* b,
    short* c
);

/**
 * Bitwise xor.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int16xor(
    const int n,
    const short* a,
    const short* b,
    short* c
);

/**
 * MSB Projection.
 *
 * @param n number of elements
 * @param original original tensor
 * @param perturbed perturbed tensor
 * @param output putput tensor
 */
void cffi_int16msbprojection(
    const int n,
    const short* original,
    const short* perturbed,
    short* output
);

/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_int16hammingdistance(
    const int n,
    const short* a,
    const short* b,
    int* dist
);

/**
 * Flip bits according to mask.
 *
 * @param n number of elements in input
 * @param mask mask array
 * @param input input array
 * @param output output array.
 */
void cffi_int16flip(
    const int n,
    const bool* mask,
    const short* input,
    short* output
);

/**
 * Set bits according to masks.
 *
 * @param n number of elements in input
 * @param set1 mask for setting to 1
 * @param set0 mask for setting to 0
 * @param input input array
 * @param output output array
 */
void cffi_int16set(
    const int n,
    const bool* set1,
    const bool* set0,
    const short* input,
    short* output
);

/**
 * Set m LSBs to zero.
 *
 * @param n number of elements in input
 * @param m number of LSBs
 * @param input input array
 * @param output output array
 */
void cffi_int16setzero(
    const int n,
    const int m,
    const short* input,
    short* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int16randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const short* input,
    short* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int16individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const short* input,
    short* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Get bits.
 *
 * @param n number of elements
 * @param input input tensor
 * @param output output tensor
 */
void cffi_int16bits(
    const int n,
    const short* input,
    bool* output
);

/**
 * Bitwise and.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int8and(
    const int n,
    const char* a,
    const char* b,
    char* c
);

/**
 * Bitwise or.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int8or(
    const int n,
    const char* a,
    const char* b,
    char* c
);

/**
 * Bitwise xor.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_int8xor(
    const int n,
    const char* a,
    const char* b,
    char* c
);

/**
 * MSB Projection.
 *
 * @param n number of elements
 * @param original original tensor
 * @param perturbed perturbed tensor
 * @param output putput tensor
 */
void cffi_int8msbprojection(
    const int n,
    const char* original,
    const char* perturbed,
    char* output
);

/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_int8hammingdistance(
    const int n,
    const char* a,
    const char* b,
    int* dist
);

/**
 * Flip bits according to mask.
 *
 * @param n number of elements in input
 * @param mask mask array
 * @param input input array
 * @param output output array.
 */
void cffi_int8flip(
    const int n,
    const bool* mask,
    const char* input,
    char* output
);

/**
 * Set bits according to masks.
 *
 * @param n number of elements in input
 * @param set1 mask for setting to 1
 * @param set0 mask for setting to 0
 * @param input input array
 * @param output output array
 */
void cffi_int8set(
    const int n,
    const bool* set1,
    const bool* set0,
    const char* input,
    char* output
);

/**
 * Set m LSBs to zero.
 *
 * @param n number of elements in input
 * @param m number of LSBs
 * @param input input array
 * @param output output array
 */
void cffi_int8setzero(
    const int n,
    const int m,
    const char* input,
    char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int8randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const char* input,
    char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_int8individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const char* input,
    char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Get bits.
 *
 * @param n number of elements
 * @param input input tensor
 * @param output output tensor
 */
void cffi_int8bits(
    const int n,
    const char* input,
    bool* output
);

/**
 * Bitwise and.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_uint8and(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
);

/**
 * Bitwise or.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_uint8or(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
);

/**
 * Bitwise xor.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param c output tensor
 */
void cffi_uint8xor(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* c
);

/**
 * MSB Projection.
 *
 * @param n number of elements
 * @param original original tensor
 * @param perturbed perturbed tensor
 * @param output putput tensor
 */
void cffi_uint8msbprojection(
    const int n,
    const unsigned char* original,
    const unsigned char* perturbed,
    unsigned char* output
);

/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_uint8hammingdistance(
    const int n,
    const unsigned char* a,
    const unsigned char* b,
    int* dist
);

/**
 * Flip bits according to mask.
 *
 * @param n number of elements in input
 * @param mask mask array
 * @param input input array
 * @param output output array.
 */
void cffi_uint8flip(
    const int n,
    const bool* mask,
    const unsigned char* input,
    unsigned char* output
);

/**
 * Set bits according to masks.
 *
 * @param n number of elements in input
 * @param set1 mask for setting to 1
 * @param set0 mask for setting to 0
 * @param input input array
 * @param output output array
 */
void cffi_uint8set(
    const int n,
    const bool* set1,
    const bool* set0,
    const unsigned char* input,
    unsigned char* output
);

/**
 * Set m LSBs to zero.
 *
 * @param n number of elements in input
 * @param m number of LSBs
 * @param input input array
 * @param output output array
 */
void cffi_uint8setzero(
    const int n,
    const int m,
    const unsigned char* input,
    unsigned char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_uint8randomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const unsigned char* input,
    unsigned char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability probability of flipping zeros
 * @param one_bit_flip_prob probability of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
void cffi_uint8individualrandomflip(
    const int n,
    const float* zero_bit_flip_prob,
    const float* one_bit_flip_prob,
    const int* protected_bits,
    const int* len_protected_bits,
    const float* rand_src,
    const unsigned char* input,
    unsigned char* output
);

/**
 * Implementation meant for Python CFFI implement random bit flips on int arrays.
 *
 * @param n number of elements in input
 * @param zero_bit_flip_probability array of probabilities of flipping zeros
 * @param one_bit_flip_prob array of probabilities of flipping ones
 * @param protected_bits array of length 32 with 1 = can flip bit, 0 = cannot flip bit
 * @param len_protected_bits length of protected_bits (should be 32)
 * @param mask array determining whether specific values in input can be changed
 * @param rand_src array with random numbers for each value
 * @param input input array
 * @param output output array.
 */
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
);

/**
 * Get bits.
 *
 * @param n number of elements
 * @param input input tensor
 * @param output output tensor
 */
void cffi_uint8bits(
    const int n,
    const unsigned char* input,
    bool* output
);