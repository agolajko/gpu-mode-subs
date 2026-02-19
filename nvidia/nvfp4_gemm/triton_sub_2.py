import functools

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _matmul_launch_metadata(grid, kernel, args):
    M, N, K = args["M"], args["N"], args["K"]
    return {
        "name": f"{kernel.name} [M={M}, N={N}, K={K}]",
        "flops": 2.0 * M * N * K,
    }


def _config(**autotune_kwargs):
    class inner:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, s):
            return functools.partial(self.fn[s], **autotune_kwargs)

    return inner

@_config(  # manual hp search lol
    # hps
    NUM_OUTER_STAGES=None,
    NUM_INNER_STAGES=None,
    WARP_SPECIALIZE_OUTER=True,
    WARP_SPECIALIZE_INNER=False,
    FLATTEN=True,
    # kernel launch
    num_warps=4,
    num_stages=2,
    num_ctas=1,  # this doesn't play nice
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_gemm_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_ptr,  # [M, N, L]
    stride_cm,
    stride_cn,  # Add this
    stride_cl,
    M,
    N,
    K,
    L,
    ELEM_PER_BYTE: tl.constexpr,
    GROUP_SZ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,  # Add this
    REP_K: tl.constexpr,
    NUM_OUTER_STAGES: tl.constexpr,
    NUM_INNER_STAGES: tl.constexpr,
    WARP_SPECIALIZE_OUTER: tl.constexpr,
    WARP_SPECIALIZE_INNER: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    output_dtype: tl.constexpr = tl.float16
    acc_dtype: tl.constexpr = tl.float32
    BLOCK_K_ELEM_PER_BYTE: tl.constexpr = BLOCK_K // ELEM_PER_BYTE
    BLOCK_K_GROUP_SZ: tl.constexpr = BLOCK_K // GROUP_SZ

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)  # Add this
    total_tiles = num_pid_m * num_pid_n * L  # Change this

    for linear in tl.range(
        pid,
        total_tiles,
        num_pid,
        num_stages=NUM_OUTER_STAGES,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_OUTER,
    ):
        # Decode linear index into (m, n, batch) tile
        pid_m = linear % num_pid_m
        pid_n = (linear // num_pid_m) % num_pid_n  # Add this
        pid_b = linear // (num_pid_m * num_pid_n)  # Change this

        # Base offsets for this tile
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N  # Add this
        offs_scale_m = pid_m * REP_M
        offs_scale_n = pid_n * REP_N  # Add this

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        for i in tl.range(
            0,
            tl.cdiv(K, BLOCK_K),
            num_stages=NUM_INNER_STAGES,
            warp_specialize=WARP_SPECIALIZE_INNER,
        ):
            offs_k = i * BLOCK_K_ELEM_PER_BYTE
            offs_scale_k = i * REP_K

            # Load A and B
            # a = a_desc.load([offs_am, pid_b, offs_k])
            # b = b_desc.load([offs_bn, pid_b, offs_k])  # Change from [0, ...] to [offs_bn, ...]
            a = a_desc.load([offs_am,offs_k, pid_b ])
            b = b_desc.load([offs_bn,offs_k, pid_b])
            a = a.reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)
            b = b.reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            # Load scale_a (unchanged)
            scale_a = (
                a_scale_desc.load([pid_b, offs_scale_m, offs_scale_k, 0, 0])
                .reshape(REP_M, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K_GROUP_SZ)
            )

            # Load scale_b - now loads the actual tile instead of always [0, ...]
            scale_b = (
                b_scale_desc.load([pid_b, offs_scale_n, offs_scale_k, 0, 0])  # Change from [pid_b, 0, ...]
                .reshape(REP_N, REP_K, 32, 4, 4)  # Change from 1 to REP_N
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_N, BLOCK_K_GROUP_SZ)  # Change from 128 to BLOCK_N
            )

            accumulator = tl.dot_scaled(
                a,  # [M, K/2]
                scale_a,  # [M, K/GROUP_SZ]
                "e2m1",
                b.T,  # [K/2, N]
                scale_b,  # [N, K/GROUP_SZ]  # No longer fixed at 128
                "e2m1",
                accumulator,
            )

        # Store C: [M, N, L]
        c_off = (
            (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_cm
            + (offs_bn + tl.arange(0, BLOCK_N))[None, :] * stride_cn  # Change
            + pid_b * stride_cl
        )
        c_mask = (
            (offs_am + tl.arange(0, BLOCK_M) < M)[:, None]
            & (offs_bn + tl.arange(0, BLOCK_N) < N)[None, :]  # Change
        )
        tl.store(c_ptr + c_off, accumulator.to(output_dtype), mask=c_mask)

def custom_kernel(data):
    a_tensor, b_tensor, _, _, sfa_permuted, sfb_permuted, c_tensor = data
    BLOCK_M = 128
    BLOCK_N = 128  # Change from 32 to 128 for GEMM
    BLOCK_K = 128
    GROUP_SZ = 16

    REP_M = BLOCK_M // 128
    REP_N = BLOCK_N // 128  # Add this
    REP_K = BLOCK_K // GROUP_SZ // 4

    SM_MULT = 1

    M, K_half, L = a_tensor.shape
    N, _, _ = b_tensor.shape  # Remove the assert, N is no longer padded
    # Remove: assert N == 128

    K = 2 * K_half
    ELEM_PER_BYTE = 2

    # TMA descriptors - same for A, but B block shape changes
    # a_tma = a_tensor.view(torch.uint8).permute(0, 2, 1)  # [M, L, K/2]
    # b_tma = b_tensor.view(torch.uint8).permute(0, 2, 1)  # [N, L, K/2]
    a_tma = a_tensor.view(torch.uint8)  # [M, L, K/2]
    b_tma = b_tensor.view(torch.uint8) # [N, L, K/2]

    a_desc = TensorDescriptor.from_tensor(
        a_tma,
        block_shape=[BLOCK_M, BLOCK_K // ELEM_PER_BYTE, 1],
    )
    b_desc = TensorDescriptor.from_tensor(
        b_tma,
        block_shape=[BLOCK_N,  BLOCK_K // ELEM_PER_BYTE, 1],  # Change BLOCK_N
    )

    # Scale descriptors
    rest_m = M // 128
    rest_n = N // 128  # No longer always 1
    rest_k = triton.cdiv(K, GROUP_SZ) // 4

    sfa_back = sfa_permuted.permute(5, 2, 4, 0, 1, 3)
    sfb_back = sfb_permuted.permute(5, 2, 4, 0, 1, 3)

    a_scale_packed = sfa_back.view(L, rest_m, rest_k, 2, 256)
    b_scale_packed = sfb_back.view(L, rest_n, rest_k, 2, 256)

    a_scale_desc = TensorDescriptor.from_tensor(
        a_scale_packed,
        block_shape=[1, REP_M, REP_K, 2, 256],
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        b_scale_packed,
        block_shape=[1, REP_N, REP_K, 2, 256],  # Change from 1 to REP_N
    )

    stride_cm, stride_cn, stride_cl = c_tensor.stride()  # (M, N, L)

    # Grid calculation - need M and N tiles now
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)  # Add this
    total_tiles = num_tiles_m * num_tiles_n * L  # Change this
    num_sms = torch.cuda.get_device_properties(a_tensor.device).multi_processor_count
    num_programs = min(total_tiles, num_sms * SM_MULT)
    grid = (num_programs,)

    kernel = block_scaled_batched_gemm_kernel[grid](  # Rename
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_tensor,
        stride_cm,
        stride_cn,  # Add this
        stride_cl,
        M,
        N,
        K,
        L,
        ELEM_PER_BYTE,
        GROUP_SZ,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        REP_M,
        REP_N,  # Add this
        REP_K,
    )

    return c_tensor