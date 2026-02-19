"""
Triton implementation for NVFP4 Group GEMM.

Computes multiple independent GEMMs with potentially different sizes:
For each group i: C[i] = A[i] @ B[i]

Based on the single GEMM pattern from nvfp4_gemm/triton_sub.py,
simplified from dual GEMM to handle group-wise computation.
"""

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


@_config(
    NUM_OUTER_STAGES=None,
    NUM_INNER_STAGES=None,
    WARP_SPECIALIZE_OUTER=True,
    WARP_SPECIALIZE_INNER=False,
    FLATTEN=True,
    num_warps=4,
    num_stages=4,
    num_ctas=1,
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_group_gemm_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_ptr,  # [M, N, L]
    stride_cm,
    stride_cn,
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
    REP_N: tl.constexpr,
    REP_K: tl.constexpr,
    NUM_OUTER_STAGES: tl.constexpr,
    NUM_INNER_STAGES: tl.constexpr,
    WARP_SPECIALIZE_OUTER: tl.constexpr,
    WARP_SPECIALIZE_INNER: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    """
    Block-scaled FP4 GEMM kernel for a single group.
    Computes: C = A @ B^T using FP4 data and FP8 scales.
    """
    output_dtype: tl.constexpr = tl.float16
    acc_dtype: tl.constexpr = tl.float32
    BLOCK_K_ELEM_PER_BYTE: tl.constexpr = BLOCK_K // ELEM_PER_BYTE
    BLOCK_K_GROUP_SZ: tl.constexpr = BLOCK_K // GROUP_SZ

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n * L

    for linear in tl.range(
        pid,
        total_tiles,
        num_pid,
        num_stages=NUM_OUTER_STAGES,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_OUTER,
    ):
        # Decode linear index into (m, n, batch) tile coordinates
        pid_m = linear % num_pid_m
        pid_n = (linear // num_pid_m) % num_pid_n
        pid_b = linear // (num_pid_m * num_pid_n)

        # Base offsets for this tile
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_scale_m = pid_m * REP_M
        offs_scale_n = pid_n * REP_N

        # Single accumulator for C = A @ B^T
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        # K-dimension loop
        for i in tl.range(
            0,
            tl.cdiv(K, BLOCK_K),
            num_stages=NUM_INNER_STAGES,
            warp_specialize=WARP_SPECIALIZE_INNER,
        ):
            offs_k = i * BLOCK_K_ELEM_PER_BYTE
            offs_scale_k = i * REP_K

            # Load A matrix tile and reshape
            a = a_desc.load([offs_am, pid_b, offs_k])
            a = a.reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)

            # Load B matrix tile and reshape
            b = b_desc.load([offs_bn, pid_b, offs_k])
            b = b.reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            # Load and transform A scale factors
            # Input: [pid_b, offs_scale_m, offs_scale_k, 2, 256]
            # Transform to: [BLOCK_M, BLOCK_K_GROUP_SZ]
            scale_a = (
                a_scale_desc.load([pid_b, offs_scale_m, offs_scale_k, 0, 0])
                .reshape(REP_M, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K_GROUP_SZ)
            )

            # Load and transform B scale factors
            # Input: [pid_b, offs_scale_n, offs_scale_k, 2, 256]
            # Transform to: [BLOCK_N, BLOCK_K_GROUP_SZ]
            scale_b = (
                b_scale_desc.load([pid_b, offs_scale_n, offs_scale_k, 0, 0])
                .reshape(REP_N, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_N, BLOCK_K_GROUP_SZ)
            )

            # Accumulate: C += A @ B^T using block-scaled FP4
            accumulator = tl.dot_scaled(
                a,        # [BLOCK_M, BLOCK_K/2]
                scale_a,  # [BLOCK_M, BLOCK_K/GROUP_SZ]
                "e2m1",   # FP4 E2M1 format
                b.T,      # [BLOCK_K/2, BLOCK_N]
                scale_b,  # [BLOCK_N, BLOCK_K/GROUP_SZ]
                "e2m1",   # FP4 E2M1 format
                accumulator,
            )

        # Store output: C[M, N, L]
        c_off = (
            (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_cm
            + (offs_bn + tl.arange(0, BLOCK_N))[None, :] * stride_cn
            + pid_b * stride_cl
        )
        c_mask = (
            (offs_am + tl.arange(0, BLOCK_M) < M)[:, None]
            & (offs_bn + tl.arange(0, BLOCK_N) < N)[None, :]
        )
        tl.store(c_ptr + c_off, accumulator.to(output_dtype), mask=c_mask)


def custom_kernel(data):
    """
    Group GEMM entry point.

    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
            - abc_tensors: List of (a, b, c) tuples, one per group
              - a: [m, k/2, 1] in torch.float4_e2m1fn_x2 (packed FP4)
              - b: [n, k/2, 1] in torch.float4_e2m1fn_x2 (packed FP4)
              - c: [m, n, 1] in torch.float16
            - sfasfb_tensors: List of (sfa, sfb) reference format tuples (unused)
            - sfasfb_reordered_tensors: List of (sfa_reord, sfb_reord) tuples
              - sfa_reord: [32, 4, rest_m, 4, rest_k, 1] in torch.float8_e4m3fn
              - sfb_reord: [32, 4, rest_n, 4, rest_k, 1] in torch.float8_e4m3fn
            - problem_sizes: List of (m, n, k, l) tuples where l=1

    Returns:
        result_tensors: List of output tensors (one per group)
    """
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    # Constants
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    GROUP_SZ = 16
    ELEM_PER_BYTE = 2
    SM_MULT = 2

    # Derived constants
    REP_M = BLOCK_M // 128
    REP_N = BLOCK_N // 128
    REP_K = BLOCK_K // GROUP_SZ // 4

    result_tensors = []

    # Process each group independently
    for group_idx in range(len(problem_sizes)):
        # Extract tensors for this group
        a_i, b_i, c_i = abc_tensors[group_idx]
        sfa_reord_i, sfb_reord_i = sfasfb_reordered_tensors[group_idx]
        m_i, n_i, k_i, l_i = problem_sizes[group_idx]

        # Actual K dimension (FP4 is packed, so shape has k/2)
        K = 2 * a_i.shape[1]

        # Setup TMA descriptors for data tensors
        # Permute to [M/N, L, K/2] for optimal TMA access pattern
        a_tma = a_i.view(torch.uint8).permute(0, 2, 1)  # [m, 1, k/2]
        b_tma = b_i.view(torch.uint8).permute(0, 2, 1)  # [n, 1, k/2]

        a_desc = TensorDescriptor.from_tensor(
            a_tma,
            block_shape=[BLOCK_M, 1, BLOCK_K // ELEM_PER_BYTE],
        )
        b_desc = TensorDescriptor.from_tensor(
            b_tma,
            block_shape=[BLOCK_N, 1, BLOCK_K // ELEM_PER_BYTE],
        )

        # Setup TMA descriptors for scale tensors
        # Transform from [32, 4, rest, 4, rest, 1] to [1, rest, rest, 2, 256]
        # Use ceiling division to match the scale tensor creation in reference.py
        rest_m = triton.cdiv(m_i, 128)
        rest_n = triton.cdiv(n_i, 128)
        rest_k = triton.cdiv(triton.cdiv(k_i, GROUP_SZ), 4)

        # Reverse the permutation from reference.py to get correct layout
        sfa_back = sfa_reord_i.permute(5, 2, 4, 0, 1, 3)  # [1, rest_m, rest_k, 32, 4, 4]
        sfb_back = sfb_reord_i.permute(5, 2, 4, 0, 1, 3)  # [1, rest_n, rest_k, 32, 4, 4]

        a_scale_packed = sfa_back.view(l_i, rest_m, rest_k, 2, 256)
        b_scale_packed = sfb_back.view(l_i, rest_n, rest_k, 2, 256)

        a_scale_desc = TensorDescriptor.from_tensor(
            a_scale_packed,
            block_shape=[1, REP_M, REP_K, 2, 256],
        )
        b_scale_desc = TensorDescriptor.from_tensor(
            b_scale_packed,
            block_shape=[1, REP_N, REP_K, 2, 256],
        )

        # Get output strides
        stride_cm, stride_cn, stride_cl = c_i.stride()

        # Calculate grid configuration
        num_tiles_m = triton.cdiv(m_i, BLOCK_M)
        num_tiles_n = triton.cdiv(n_i, BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n * l_i

        # Limit number of programs to avoid oversubscription
        num_sms = torch.cuda.get_device_properties(a_i.device).multi_processor_count
        num_programs = min(total_tiles, num_sms * SM_MULT)
        grid = (num_programs,)

        # Launch kernel for this group
        block_scaled_group_gemm_kernel[grid](
            a_desc,
            a_scale_desc,
            b_desc,
            b_scale_desc,
            c_i,
            stride_cm,
            stride_cn,
            stride_cl,
            m_i,
            n_i,
            k_i,
            l_i,
            ELEM_PER_BYTE,
            GROUP_SZ,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            REP_M,
            REP_N,
            REP_K,
        )

        result_tensors.append(c_i)

    return result_tensors
