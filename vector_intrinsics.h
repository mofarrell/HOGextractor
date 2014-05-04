// Created by Michael and Bram
//

//#define AVX
//#define DOUBLE

#ifndef AVX
// Include SSE headers
#include "smmintrin.h"
#include "xmmintrin.h"
#include "emmintrin.h"

#ifndef DOUBLE
////////////////////////////
// SSE with SINGLE PRECISION
////////////////////////////
#define SIMD_WIDTH 4

// Typedefs
typedef __m128 vreal;
typedef float real;
#define mxCLASS mxSINGLE_CLASS
typedef __m128i vnat;
typedef int nat;

typedef __m128 vmask;

// Real operations
#define set_real(v) _mm_set_ps1((v))
#define set_vreal(v0, v1, v2, v3) _mm_set_ps((v3), (v2), (v1), (v0))
#define load_real(p) _mm_load_ps1((p))
#define load_vreal(p) _mm_loadu_ps((p))
#define store_vreal(p, v) _mm_storeu_ps((p), (v))
#define add_vreal(v1, v2) _mm_add_ps((v1), (v2))
#define sub_vreal(v1, v2) _mm_sub_ps((v1), (v2))
#define mul_vreal(v1, v2) _mm_mul_ps((v1), (v2))
#define div_vreal(v1, v2) _mm_div_ps((v1), (v2))
#define floor_vreal(v) _mm_floor_ps((v))
#define sqrt_vreal(v) _mm_sqrt_ps((v))
#define or_vreal(v1, v2) _mm_or_ps((v1), (v2))
#define and_vreal(m, v) _mm_and_ps((m), (v))
#define andnot_vreal(m, v) _mm_andnot_ps((m), (v))
#define cmpgt_vreal(v1, v2) _mm_cmpgt_ps((v1), (v2))

// Type casts and conversion
#define vnat_to_vreal(v) _mm_castsi128_ps((v))
#define vreal_to_vnat(v) _mm_castps_si128((v))
#define vnat_convertto_vreal(v) _mm_cvtepi32_ps((v))
#define vreal_convertto_vnat(v) _mm_cvtps_epi32((v))

// Natural operations
#define set_nat(v) _mm_set1_epi32((v))
#define set_vnat(v0, v1, v2, v3) _mm_set_epi32((v3), (v2), (v1), (v0))
#define load_vnat(p) _mm_loadu_si128((p))
#define store_vnat(p, v) _mm_storeu_si128((vnat *)(p), (v))
#define add_vnat(v1, v2) _mm_add_epi32((v1), (v2))
#define sub_vnat(v1, v2) _mm_sub_epi32((v1), (v2))
#define mul_vnat(v1, v2) _mm_mul_epi32((v1), (v2))
#define div_vnat(v1, v2) _mm_div_epi32((v1), (v2))
#define or_vnat(v1, v2) _mm_or_si128((v1), (v2))
#define and_vnat(m, v) _mm_and_si128(vreal_to_vnat((m)), (v))
#define andnot_vnat(m, v) _mm_andnot_si128(vreal_to_vnat((m)), (v))

// Constants
const vreal SIGNMASK = vnat_to_vreal(set_nat(0x80000000));
#define neg_vreal(v) _mm_xor_ps((v), (SIGNMASK))
const vreal SIMD_WIDTH_IDX_REAL = set_vreal(0.0f, 1.0f, 2.0f, 3.0f);
const vnat SIMD_WIDTH_IDX_NAT = set_vnat(0, 1, 2, 3);

#else
////////////////////////////
// SSE with DOUBLE PRECISION
////////////////////////////
#define SIMD_WIDTH 2

// Typedefs
typedef __m128d vreal;
typedef double real;
#define mxCLASS mxDOUBLE_CLASS
typedef __m128i vnat;
typedef long nat;

typedef __m128d vmask;

// Real operations
#define set_real(v) _mm_set_pd1((v))
#define set_vreal(v0, v1) _mm_set_pd((v1), (v0))
#define load_real(p) _mm_load_pd1((p))
#define load_vreal(p) _mm_loadu_pd((p))
#define store_vreal(p, v) _mm_storeu_pd((p), (v))
#define add_vreal(v1, v2) _mm_add_pd((v1), (v2))
#define sub_vreal(v1, v2) _mm_sub_pd((v1), (v2))
#define mul_vreal(v1, v2) _mm_mul_pd((v1), (v2))
#define div_vreal(v1, v2) _mm_div_pd((v1), (v2))
#define floor_vreal(v) _mm_floor_pd((v))
#define sqrt_vreal(v) _mm_sqrt_pd((v))
#define or_vreal(v1, v2) _mm_or_pd((v1), (v2))
#define and_vreal(m, v) _mm_and_pd((m), (v))
#define andnot_vreal(m, v) _mm_andnot_pd((m), (v))
#define cmpgt_vreal(v1, v2) _mm_cmpgt_pd((v1), (v2))

// Type casts and conversion
#define vnat_to_vreal(v) _mm_castsi128_pd((v))
#define vreal_to_vnat(v) _mm_castpd_si128((v))
#define vnat_convertto_vreal(v) _mm_cvtepi32_pd(_mm_cvtepi64_epi32((v)))
#define vreal_convertto_vnat(v) _mm_cvtepi32_epi64(_mm_cvtpd_epi32((v)))

// Natural operations
#define set_nat(v) _mm_set1_epi64((__m64)(nat)(v))
#define set_vnat(v0, v1) _mm_set_epi64((__m64)(nat)(v1), (__m64)(nat)(v0))
#define load_vnat(p) _mm_loadu_si128((__m64 *)(p))
#define store_vnat(p, v) _mm_storeu_si128((vnat *)(p), (v))
#define add_vnat(v1, v2) _mm_add_epi64((v1), (v2))
#define sub_vnat(v1, v2) _mm_sub_epi64((v1), (v2))
#define mul_vnat(v1, v2) _mm_mul_epi64((v1), (v2))
#define div_vnat(v1, v2) _mm_div_epi64((v1), (v2))
#define or_vnat(v1, v2) _mm_or_si128((v1), (v2))
#define and_vnat(m, v) _mm_and_si128(vreal_to_vnat((m)), (v))
#define andnot_vnat(m, v) _mm_andnot_si128(vreal_to_vnat((m)), (v))

// Constants
const vreal SIGNMASK = vnat_to_vreal(set_nat(0x8000000000000000L));
#define neg_vreal(v) _mm_xor_pd((v), (SIGNMASK))
const vreal SIMD_WIDTH_IDX_REAL = set_vreal(0.0, 1.0);
const vnat SIMD_WIDTH_IDX_NAT = set_vnat(0L, 1L);

#endif
#else
// Include AVX headers
#include "immintrin.h"

#ifndef DOUBLE
////////////////////////////
// AVX with SINGLE PRECISION
////////////////////////////
#define SIMD_WIDTH 8

// Typedefs
typedef __m256 vreal;
typedef float real;
#define mxCLASS mxSINGLE_CLASS
typedef __m256i vnat;
typedef int nat;

typedef __m256 vmask;

// Real operations
#define set_real(v) _mm256_set_ps1((v))
#define set_vreal(v0, v1, v2, v3, v4, v5, v6, v7) \
    _mm256_set_ps((v7), (v6), (v5), (v4), (v3), (v2), (v1), (v0))
#define load_real(p) _mm256_load_ps1((p))
#define load_vreal(p) _mm256_loadu_ps((p))
#define store_vreal(p, v) _mm256_storeu_ps((p), (v))
#define add_vreal(v1, v2) _mm256_add_ps((v1), (v2))
#define sub_vreal(v1, v2) _mm256_sub_ps((v1), (v2))
#define mul_vreal(v1, v2) _mm256_mul_ps((v1), (v2))
#define div_vreal(v1, v2) _mm256_div_ps((v1), (v2))
#define floor_vreal(v) _mm256_floor_ps((v))
#define sqrt_vreal(v) _mm256_sqrt_ps((v))
#define or_vreal(v1, v2) _mm256_or_ps((v1), (v2))
#define and_vreal(m, v) _mm256_and_ps((m), (v))
#define andnot_vreal(m, v) _mm256_andnot_ps((m), (v))
#define cmpgt_vreal(v1, v2) _mm256_cmpgt_ps((v1), (v2))

// Type casts and conversion
#define vnat_to_vreal(v) _mm256_castsi256_ps((v))
#define vreal_to_vnat(v) _mm256_castps_si256((v))
#define vnat_convertto_vreal(v) _mm256_cvtepi32_ps((v))
#define vreal_convertto_vnat(v) _mm256_cvtps_epi32((v))

// Natural operations
#define set_nat(v) _mm256_set1_epi32((v))
#define set_vnat(v0, v1, v2, v3, v4, v5, v6, v7) \
  _mm256_set_epi32((v7), (v6), (v5), (v4), (v3), (v2), (v1), (v0))
#define load_vnat(p) _mm256_loadu_si256((p))
#define store_vnat(p, v) _mm256_storeu_si256((vnat *)(p), (v))
#define add_vnat(v1, v2) _mm256_add_epi32((v1), (v2))
#define sub_vnat(v1, v2) _mm256_sub_epi32((v1), (v2))
#define mul_vnat(v1, v2) _mm256_mul_epi32((v1), (v2))
#define div_vnat(v1, v2) _mm256_div_epi32((v1), (v2))
#define or_vnat(v1, v2) _mm256_or_si256((v1), (v2))
#define and_vnat(m, v) _mm256_and_si256(vreal_to_vnat((m)), (v))
#define andnot_vnat(m, v) _mm256_andnot_si256(vreal_to_vnat((m)), (v))

// Constants
const vreal SIGNMASK = vnat_to_vreal(set_nat(0x80000000));
#define neg_vreal(v) _mm256_xor_ps((v), (SIGNMASK))
const vreal SIMD_WIDTH_IDX_REAL = set_vreal(0.0f, 1.0f, 2.0f, 3.0f,
                                            4.0f, 5.0f, 6.0f, 7.0f);
const vnat SIMD_WIDTH_IDX_NAT = set_vnat(0, 1, 2, 3, 4, 5, 6, 7);

#else
////////////////////////////
// AVX with DOUBLE PRECISION
////////////////////////////
#define SIMD_WIDTH 2

// Typedefs
typedef __m128d vreal;
typedef double real;
#define mxCLASS mxDOUBLE_CLASS
typedef __m128i vnat;
typedef long nat;

typedef __m128d vmask;

// Real operations
#define set_real(v) _mm256_set_pd1((v))
#define set_vreal(v0, v1, v2, v3) _mm256_set_pd((v3), (v2), (v1), (v0))
#define load_real(p) _mm256_load_pd1((p))
#define load_vreal(p) _mm256_loadu_pd((p))
#define store_vreal(p, v) _mm256_storeu_pd((p), (v))
#define add_vreal(v1, v2) _mm256_add_pd((v1), (v2))
#define sub_vreal(v1, v2) _mm256_sub_pd((v1), (v2))
#define mul_vreal(v1, v2) _mm256_mul_pd((v1), (v2))
#define div_vreal(v1, v2) _mm256_div_pd((v1), (v2))
#define floor_vreal(v) _mm256_floor_pd((v))
#define sqrt_vreal(v) _mm256_sqrt_pd((v))
#define or_vreal(v1, v2) _mm256_or_pd((v1), (v2))
#define and_vreal(m, v) _mm256_and_pd((m), (v))
#define andnot_vreal(m, v) _mm256_andnot_pd((m), (v))
#define cmpgt_vreal(v1, v2) _mm256_cmpgt_pd((v1), (v2))

// Type casts and conversion
#define vnat_to_vreal(v) _mm256_castsi256_pd((v))
#define vreal_to_vnat(v) _mm256_castpd_si256((v))
#define vnat_convertto_vreal(v) _mm256_cvtepi32_pd(_mm256_cvtepi64_epi32((v)))
#define vreal_convertto_vnat(v) _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32((v)))

// Natural operations
#define set_nat(v) _mm256_set1_epi64x((__m64)(nat)(v))
#define set_vnat(v0, v1, v2, v3) \
    _mm256_set_epi64((__m64)(nat)(v3), (__m64)(nat)(v2), \
                     (__m64)(nat)(v1), (__m64)(nat)(v0))
#define load_vnat(p) _mm256_loadu_si256((__m64 *)(p))
#define store_vnat(p, v) _mm256_storeu_si256((vnat *)(p), (v))
#define add_vnat(v1, v2) _mm256_add_epi64((v1), (v2))
#define sub_vnat(v1, v2) _mm256_sub_epi64((v1), (v2))
#define mul_vnat(v1, v2) _mm256_mul_epi64((v1), (v2))
#define div_vnat(v1, v2) _mm_div_epi64((v1), (v2))
#define or_vnat(v1, v2) _mm256_or_si256((v1), (v2))
#define and_vnat(m, v) _mm256_and_si256(vreal_to_vnat((m)), (v))
#define andnot_vnat(m, v) _mm256_andnot_si256(vreal_to_vnat((m)), (v))

// Constants
const vreal SIGNMASK = vnat_to_vreal(set_nat(0x8000000000000000L));
#define neg_vreal(v) _mm256_xor_pd((v), (SIGNMASK))
const vreal SIMD_WIDTH_IDX_REAL = set_vreal(0.0, 1.0, 2.0, 3.0);
const vnat SIMD_WIDTH_IDX_NAT = set_vnat(0L, 1L, 2L, 3L);

#endif
#endif





