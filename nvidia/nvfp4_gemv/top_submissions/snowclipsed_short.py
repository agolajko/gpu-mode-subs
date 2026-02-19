import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = """
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// 32-byte aligned type for 256-bit loads
struct __align__(32) uint8_256bit { uint4 lo; uint4 hi; };

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void cvt_f4x8_to_f16x8(uint32_t src, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
    asm volatile(
        "{ .reg .b8 b0, b1, b2, b3; "
        "mov.b32 {b0, b1, b2, b3}, %4; "
        "cvt.rn.f16x2.e2m1x2 %0, b0; "
        "cvt.rn.f16x2.e2m1x2 %1, b1; "
        "cvt.rn.f16x2.e2m1x2 %2, b2; "
        "cvt.rn.f16x2.e2m1x2 %3, b3; }"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(src)
    );
}

__device__ __forceinline__ uint32_t cvt_f8x2_to_f16x2(uint16_t src) {
    uint32_t dst;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
    return dst;
}

__device__ __forceinline__ half2 process_16_elements(uint32_t a_lo, uint32_t a_hi, uint32_t b_lo, uint32_t b_hi, half2 scale_h2) {
    uint32_t a0, a1, a2, a3, a4, a5, a6, a7;
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
    cvt_f4x8_to_f16x8(a_lo, a0, a1, a2, a3);
    cvt_f4x8_to_f16x8(a_hi, a4, a5, a6, a7);
    cvt_f4x8_to_f16x8(b_lo, b0, b1, b2, b3);
    cvt_f4x8_to_f16x8(b_hi, b4, b5, b6, b7);
    
    half2 ab0 = __hmul2(*reinterpret_cast<half2*>(&a0), *reinterpret_cast<half2*>(&b0));
    half2 ab1 = __hmul2(*reinterpret_cast<half2*>(&a1), *reinterpret_cast<half2*>(&b1));
    half2 ab2 = __hmul2(*reinterpret_cast<half2*>(&a2), *reinterpret_cast<half2*>(&b2));
    half2 ab3 = __hmul2(*reinterpret_cast<half2*>(&a3), *reinterpret_cast<half2*>(&b3));
    half2 ab4 = __hmul2(*reinterpret_cast<half2*>(&a4), *reinterpret_cast<half2*>(&b4));
    half2 ab5 = __hmul2(*reinterpret_cast<half2*>(&a5), *reinterpret_cast<half2*>(&b5));
    half2 ab6 = __hmul2(*reinterpret_cast<half2*>(&a6), *reinterpret_cast<half2*>(&b6));
    half2 ab7 = __hmul2(*reinterpret_cast<half2*>(&a7), *reinterpret_cast<half2*>(&b7));
    
    half2 sum01 = __hadd2(ab0, ab1);
    half2 sum23 = __hadd2(ab2, ab3);
    half2 sum45 = __hadd2(ab4, ab5);
    half2 sum67 = __hadd2(ab6, ab7);
    half2 sum0123 = __hadd2(sum01, sum23);
    half2 sum4567 = __hadd2(sum45, sum67);
    half2 local_sum = __hadd2(sum0123, sum4567);
    
    return __hmul2(local_sum, scale_h2);
}

__global__ void gemv_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA,
    const uint8_t* __restrict__ SFB,
    half* __restrict__ C,
    int M, int K, int L,
    int64_t a_s0, int64_t a_s2, int64_t b_s2,
    int64_t sfa_s0, int64_t sfa_s2,
    int64_t sfb_s2,
    int64_t c_s0, int64_t c_s2
) {
    constexpr int WARPS_PER_BLOCK = 4;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int batch = blockIdx.y;
    
    if (row >= M) return;
    
    const uint8_t* sfa_row = SFA + row * sfa_s0 + batch * sfa_s2;
    const uint8_t* sfb_batch = SFB + batch * sfb_s2;
    const uint8_t* A_row = A + row * a_s0 + batch * a_s2;
    const uint8_t* B_batch = B + batch * b_s2;
    
    half2 acc_h2 = __float2half2_rn(0.0f);
    int K_scales = K / 16;
    
    // Process 4 scale groups (64 FP4 elements = 32 bytes) per lane per iteration
    // 32 lanes * 64 elements = 2048 elements per iteration
    for (int scale_base = 0; scale_base < K_scales; scale_base += 128) {
        int lane_scale_idx = scale_base + lane_id * 4;
        
        if (lane_scale_idx + 3 >= K_scales) {
            // Tail handling - fall back to smaller loads
            for (int s = lane_scale_idx; s < K_scales && s < lane_scale_idx + 4; s += 2) {
                if (s + 1 >= K_scales) {
                    if (s < K_scales) {
                        uint8_t sfa0 = sfa_row[s];
                        uint8_t sfb0 = sfb_batch[s];
                        uint16_t sf_packed = (uint16_t(sfb0) << 8) | uint16_t(sfa0);
                        uint32_t sf_f16x2 = cvt_f8x2_to_f16x2(sf_packed);
                        half2 sf_h2 = *reinterpret_cast<half2*>(&sf_f16x2);
                        half scale = __hmul(sf_h2.x, sf_h2.y);
                        half2 scale_h2 = __half2half2(scale);
                        
                        int k_byte = s * 8;
                        uint2 a_vec = *reinterpret_cast<const uint2*>(A_row + k_byte);
                        uint2 b_vec = *reinterpret_cast<const uint2*>(B_batch + k_byte);
                        acc_h2 = __hadd2(acc_h2, process_16_elements(a_vec.x, a_vec.y, b_vec.x, b_vec.y, scale_h2));
                    }
                    break;
                }
                uint16_t sfa_pair = *reinterpret_cast<const uint16_t*>(sfa_row + s);
                uint16_t sfb_pair = *reinterpret_cast<const uint16_t*>(sfb_batch + s);
                uint8_t sfa0 = sfa_pair & 0xFF, sfa1 = sfa_pair >> 8;
                uint8_t sfb0 = sfb_pair & 0xFF, sfb1 = sfb_pair >> 8;
                
                uint16_t sf_packed0 = (uint16_t(sfb0) << 8) | uint16_t(sfa0);
                uint16_t sf_packed1 = (uint16_t(sfb1) << 8) | uint16_t(sfa1);
                uint32_t sf0_f16x2 = cvt_f8x2_to_f16x2(sf_packed0);
                uint32_t sf1_f16x2 = cvt_f8x2_to_f16x2(sf_packed1);
                half2 sf0_h2 = *reinterpret_cast<half2*>(&sf0_f16x2);
                half2 sf1_h2 = *reinterpret_cast<half2*>(&sf1_f16x2);
                half scale0 = __hmul(sf0_h2.x, sf0_h2.y);
                half scale1 = __hmul(sf1_h2.x, sf1_h2.y);
                half2 scale0_h2 = __half2half2(scale0);
                half2 scale1_h2 = __half2half2(scale1);
                
                int k_byte = s * 8;
                uint4 a_vec = *reinterpret_cast<const uint4*>(A_row + k_byte);
                uint4 b_vec = *reinterpret_cast<const uint4*>(B_batch + k_byte);
                acc_h2 = __hadd2(acc_h2, process_16_elements(a_vec.x, a_vec.y, b_vec.x, b_vec.y, scale0_h2));
                acc_h2 = __hadd2(acc_h2, process_16_elements(a_vec.z, a_vec.w, b_vec.z, b_vec.w, scale1_h2));
            }
            break;
        }
        
        // Load 4 scale factors at once (32-bit load = 4 FP8 scales)
        uint32_t sfa_quad = *reinterpret_cast<const uint32_t*>(sfa_row + lane_scale_idx);
        uint32_t sfb_quad = *reinterpret_cast<const uint32_t*>(sfb_batch + lane_scale_idx);
        
        // Extract as packed pairs: lo 16 bits = scales 0,1; hi 16 bits = scales 2,3
        uint16_t sfa_01 = sfa_quad & 0xFFFF;
        uint16_t sfa_23 = sfa_quad >> 16;
        uint16_t sfb_01 = sfb_quad & 0xFFFF;
        uint16_t sfb_23 = sfb_quad >> 16;
        
        // Convert packed FP8x2 to FP16x2 directly
        uint32_t sfa_01_f16 = cvt_f8x2_to_f16x2(sfa_01);
        uint32_t sfa_23_f16 = cvt_f8x2_to_f16x2(sfa_23);
        uint32_t sfb_01_f16 = cvt_f8x2_to_f16x2(sfb_01);
        uint32_t sfb_23_f16 = cvt_f8x2_to_f16x2(sfb_23);
        
        // Multiply scale pairs and broadcast
        half2 sfa_01_h2 = *reinterpret_cast<half2*>(&sfa_01_f16);
        half2 sfa_23_h2 = *reinterpret_cast<half2*>(&sfa_23_f16);
        half2 sfb_01_h2 = *reinterpret_cast<half2*>(&sfb_01_f16);
        half2 sfb_23_h2 = *reinterpret_cast<half2*>(&sfb_23_f16);
        
        half2 sf_prod_01 = __hmul2(sfa_01_h2, sfb_01_h2);
        half2 sf_prod_23 = __hmul2(sfa_23_h2, sfb_23_h2);
        
        half2 scale0_h2 = __half2half2(sf_prod_01.x);
        half2 scale1_h2 = __half2half2(sf_prod_01.y);
        half2 scale2_h2 = __half2half2(sf_prod_23.x);
        half2 scale3_h2 = __half2half2(sf_prod_23.y);
        
        // 256-bit load: 32 bytes = 64 FP4 elements = 4 scale groups
        int k_byte = lane_scale_idx * 8;
        const uint8_256bit* a_ptr256 = reinterpret_cast<const uint8_256bit*>(A_row + k_byte);
        const uint8_256bit* b_ptr256 = reinterpret_cast<const uint8_256bit*>(B_batch + k_byte);
        
        uint8_256bit a_data = *a_ptr256;
        uint8_256bit b_data = *b_ptr256;
        
        // Process 4 groups of 16 elements each
        acc_h2 = __hadd2(acc_h2, process_16_elements(a_data.lo.x, a_data.lo.y, b_data.lo.x, b_data.lo.y, scale0_h2));
        acc_h2 = __hadd2(acc_h2, process_16_elements(a_data.lo.z, a_data.lo.w, b_data.lo.z, b_data.lo.w, scale1_h2));
        acc_h2 = __hadd2(acc_h2, process_16_elements(a_data.hi.x, a_data.hi.y, b_data.hi.x, b_data.hi.y, scale2_h2));
        acc_h2 = __hadd2(acc_h2, process_16_elements(a_data.hi.z, a_data.hi.w, b_data.hi.z, b_data.hi.w, scale3_h2));
    }
    
    float acc = __half2float(acc_h2.x) + __half2float(acc_h2.y);
    acc = warp_reduce_sum(acc);
    
    if (lane_id == 0) {
        C[row * c_s0 + batch * c_s2] = __float2half(acc);
    }
}

torch::Tensor gemv_cuda(torch::Tensor a, torch::Tensor b,
                        torch::Tensor sfa, torch::Tensor sfb,
                        torch::Tensor c) {
    int M = a.size(0), K = a.size(1) * 2, L = a.size(2);
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, L);
    dim3 block(32 * WARPS_PER_BLOCK);
    
    gemv_kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t*>(a.data_ptr()),
        reinterpret_cast<const uint8_t*>(b.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfb.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, K, L,
        a.stride(0), a.stride(2), b.stride(2),
        sfa.stride(0), sfa.stride(2),
        sfb.stride(2),
        c.stride(0), c.stride(2)
    );
    return c;
}
"""

cpp_source = "torch::Tensor gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);"

module = load_inline(
    name='gemv_256bit',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gemv_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17', '--generate-code=arch=compute_100a,code=sm_100a'],
    verbose=True
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, sfa_perm, sfb_perm, c = data
    return module.gemv_cuda(a, b, sfa, sfb, c)