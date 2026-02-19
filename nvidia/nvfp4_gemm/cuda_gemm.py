import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import os
import hashlib

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define SF_BLOCK_SIZE 16  // Each scale factor covers 16 FP4 elements

// FP4 E2M1 decoding lookup table (4-bit float to float32)
__device__ const float fp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Extract and decode a single FP4 value from packed uint8
__device__ __forceinline__ float decode_fp4(uint8_t packed, int idx) {
    // idx=0 gets lower 4 bits, idx=1 gets upper 4 bits
    uint8_t val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return fp4_e2m1_lut[val];
}

// FP8 E4M3 to float32 conversion
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = x;
    return float(fp8_val);
}

__global__ void fp4_gemv_kernel_batched(
    const uint8_t* __restrict__ A,      // M x (K/2) x L
    const uint8_t* __restrict__ x,      // 1 x (K/2) x L
    const uint8_t* __restrict__ scale_a,  // M x (K/16) x L
    const uint8_t* __restrict__ scale_x,  // 1 x (K/16) x L
    __half* __restrict__ y,             // M x 1 x L
    const int M,
    const int K,
    const int K_packed,
    const int L
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Add batch dimension to grid
    const int row = blockIdx.x * warps_per_block + warp_id;
    const int batch = blockIdx.y;  // NEW: batch index
    
    if (row >= M || batch >= L) return;
    
    const int num_scale_blocks = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
    
    // Shared memory per batch
    extern __shared__ uint8_t smem[];
    uint8_t* smem_x = smem;
    uint8_t* smem_scale_x = smem + K_packed;
    
    // Offset pointers by batch
    const int64_t batch_offset_packed = (int64_t)M * K_packed * batch;
    const int64_t batch_offset_scale = (int64_t)M * num_scale_blocks * batch;
    const int64_t x_batch_offset = (int64_t)K_packed * batch;
    const int64_t scale_x_batch_offset = (int64_t)num_scale_blocks * batch;
    
    // Load input vector for this batch
    for (int i = tid; i < K_packed; i += blockDim.x) {
        smem_x[i] = x[x_batch_offset + i];
    }
    for (int i = tid; i < num_scale_blocks; i += blockDim.x) {
        smem_scale_x[i] = scale_x[scale_x_batch_offset + i];
    }
    
    __syncthreads();
    
    // Compute for this row and batch
    const uint8_t* A_row = A + batch_offset_packed + (int64_t)row * K_packed;
    const uint8_t* scale_a_row = scale_a + batch_offset_scale + (int64_t)row * num_scale_blocks;
    
    float sum = 0.0f;
    
    for (int k_packed = lane_id; k_packed < K_packed; k_packed += WARP_SIZE) {
        for (int sub_idx = 0; sub_idx < 2; sub_idx++) {
            int k = k_packed * 2 + sub_idx;
            if (k >= K) break;
            
            int scale_idx = k / SF_BLOCK_SIZE;
            
            float a_fp4 = decode_fp4(A_row[k_packed], sub_idx);
            float x_fp4 = decode_fp4(smem_x[k_packed], sub_idx);
            
            float sa = fp8_e4m3_to_float(scale_a_row[scale_idx]);
            float sx = fp8_e4m3_to_float(smem_scale_x[scale_idx]);
            
            sum += a_fp4 * sa * x_fp4 * sx;
        }
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    if (lane_id == 0) {
        y[batch * M + row] = __float2half(sum);
    }
}

torch::Tensor fp4_gemv_forward_batched(
    torch::Tensor A,
    torch::Tensor x,
    torch::Tensor scale_a,
    torch::Tensor scale_x,
    int M,
    int K,
    int K_packed,
    int L
) {
    // Output shape: [L * M] which will be reshaped to [L, M] in Python
    auto y = torch::zeros({L * M}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
    
    const int warps_per_block = 8;  // Increased from 4 for better occupancy
    const int threads = warps_per_block * WARP_SIZE;
    const int blocks_m = (M + warps_per_block - 1) / warps_per_block;
    
    // 2D grid: (M_blocks, L)
    dim3 grid(blocks_m, L);
    dim3 block(threads);
    
    const int num_scale_blocks = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
    size_t smem_size = K_packed * sizeof(uint8_t) + num_scale_blocks * sizeof(uint8_t);
    
    fp4_gemv_kernel_batched<<<grid, block, smem_size>>>(
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        reinterpret_cast<const uint8_t*>(x.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale_a.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale_x.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M,
        K,
        K_packed,
        L
    );
    
    return y;
}
"""


cpp_source = """
torch::Tensor fp4_gemv_forward_batched(
    torch::Tensor A,
    torch::Tensor x,
    torch::Tensor scale_a,
    torch::Tensor scale_x,
    int M,
    int K,
    int K_packed,
    int L
);
"""

# Compile the extension
fp4_gemv_module = None

def get_module():
    global fp4_gemv_module
    if fp4_gemv_module is None:
        # Create a cache directory
        cache_dir = os.path.expanduser("~/.cache/torch_extensions")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a hash of the source to detect changes
        source_hash = hashlib.md5((cuda_source + cpp_source).encode()).hexdigest()[:8]
        
        fp4_gemv_module = load_inline(
            name=f'fp4_gemv_{source_hash}',  # Unique name per version
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['fp4_gemv_forward_batched'],
            extra_cuda_cflags=['-O3', '--use_fast_math','-arch=sm_100','--maxrregcount=255', '-Xptxas=-v'],
            build_directory=cache_dir,  # Persistent cache
            verbose=False  # Set to True for debugging
        )
    return fp4_gemv_module


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format for torch._scaled_mm"""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    
    return rearranged.flatten()


def custom_kernel(data: input_t) -> output_t:
    """
    CUDA-accelerated FP4 block-scaled GEMV kernel with batch fusion.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    # Original shapes: a_ref=[M, K_packed, L], b_ref=[1, K_packed, L]
    M, K_packed, L = a_ref.shape
    K = K_packed * 2
    
    module = get_module()
    
    # Move scales to GPU once
    sfa_gpu = sfa_ref_cpu.cuda()
    sfb_gpu = sfb_ref_cpu.cuda()
    
    # Convert to uint8 FIRST, then reshape
    # Original: a_ref[M, K_packed, L] -> view as uint8 -> permute(2,0,1) -> [L, M, K_packed]
    A_all = a_ref.view(torch.uint8).permute(2, 0, 1).contiguous()
    
    # Original: b_ref[1, K_packed, L] -> view as uint8 -> [K_packed, L] -> permute -> [L, K_packed]
    x_all = b_ref.view(torch.uint8)[0, :, :].permute(1, 0).contiguous()
    
    # Scale factors: sfa_ref_cpu[M, K/16, L] -> [L, M, K/16]
    scale_a_all = sfa_gpu.view(torch.uint8).permute(2, 0, 1).contiguous()
    
    # Scale factors: sfb_ref_cpu[1, K/16, L] -> [K/16, L] -> permute -> [L, K/16]
    scale_x_all = sfb_gpu.view(torch.uint8)[0, :, :].permute(1, 0).contiguous()
    
    # Single kernel launch for all batches
    result = module.fp4_gemv_forward_batched(
        A_all,
        x_all,
        scale_a_all,
        scale_x_all,
        M, K, K_packed, L
    )
    
    # result is [L*M], reshape to [L, M] then transpose to [M, L]
    c_ref[:, 0, :] = result.view(L, M).t()
    
    return c_ref