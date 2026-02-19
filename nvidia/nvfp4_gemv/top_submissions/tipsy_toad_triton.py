"""
Kernel for Nvidia's Batched NVFP4 GEMV competition:
https://www.gpumode.com/v2/leaderboard/595?tab=rankings

Currently 7th with: 25.316μs

1st: 21.691μs | +3.625μs

k: 16384; l: 1; m: 7168; seed: 1111
 ⏱ 27.0 ± 0.05 µs
 ⚡ 26.5 µs 🐌 28.7 µs

k: 7168; l: 8; m: 4096; seed: 1111
 ⏱ 36.6 ± 0.05 µs
 ⚡ 34.8 µs 🐌 38.0 µs

k: 2048; l: 4; m: 7168; seed: 1111
 ⏱ 16.4 ± 0.01 µs
 ⚡ 16.3 µs 🐌 16.4 µs
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


def _get_configs():  # TODO: autotune when you get resources
    return [
        triton.Config(
            dict(
                NUM_OUTER_STAGES=None,
                NUM_INNER_STAGES=None,
                WARP_SPECIALIZE_OUTER=False,
                WARP_SPECIALIZE_INNER=False,
                FLATTEN=True,
            ),
            num_warps=w,
            num_stages=s,
            num_ctas=1,
        )
        for w in [4]
        for s in [6]
        # for w in [2, 4, 6, 8]
        # for s in [2, 4, 5, 6, 7, 8]
        # for c in [1, 2]
    ]


# @triton.autotune(configs=_get_configs(), key=["M", "N", "K", "L"], cache_results=True)
@_config(  # manual hp search lol
    # hps
    NUM_OUTER_STAGES=None,
    NUM_INNER_STAGES=None,
    WARP_SPECIALIZE_OUTER=True,
    WARP_SPECIALIZE_INNER=False,
    FLATTEN=True,
    # kernel launch
    num_warps=4,
    num_stages=4,
    num_ctas=1,  # this doesn't play nice
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_gemv_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_ptr,  # [M, L]
    stride_cm,
    stride_cl,
    M,
    N,
    K,
    L,
    ELEM_PER_BYTE: tl.constexpr,  # 2 for nvfp4
    GROUP_SZ: tl.constexpr,  # 16 for nvfp4
    # Block tuning
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REP_M: tl.constexpr,  # BLOCK_M / 128
    REP_K: tl.constexpr,  # BLOCK_K / GROUP_SZ / 4
    # Kernel tuning
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
    total_tiles = num_pid_m * L  # all (m, batch) tiles

    for linear in tl.range(
        pid,
        total_tiles,
        num_pid,
        num_stages=NUM_OUTER_STAGES,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_OUTER,
    ):
        pid_m = linear % num_pid_m
        pid_b = linear // num_pid_m

        # Base offsets for this tile
        offs_am = pid_m * BLOCK_M
        offs_scale_m = pid_m * REP_M

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        for i in tl.range(
            0,
            tl.cdiv(K, BLOCK_K),
            num_stages=NUM_INNER_STAGES,
            warp_specialize=WARP_SPECIALIZE_INNER,
        ):
            offs_k = i * BLOCK_K_ELEM_PER_BYTE
            offs_scale_k = i * REP_K

            # A: [M, L, K/2]; block_shape [BLOCK_M, 1, BLOCK_K/2]
            # B: [1, L, K/2]; block_shape [BLOCK_N, 1, BLOCK_K/2] (padded)
            a = a_desc.load([offs_am, pid_b, offs_k])
            b = b_desc.load([0, pid_b, offs_k])
            a = a.reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)  # squeeze dim 1
            b = b.reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            # scale_a: [L, rest_m, rest_k, 2, 256]
            # => [rep_m, rep_k, x, scale_m, scale_k] (x=32, scale_m=4, scale_k=4)
            # => [rep_m, scale_m, x, rep_k, scale_k]
            # => [(rep_m, scale_m, x), (rep_k scale_k)] (BLOCK_M, BLOCK_K / GROUP_SZ)
            # Belive it or not, this a the contiguous layout (I think).
            scale_a = (
                a_scale_desc.load([pid_b, offs_scale_m, offs_scale_k, 0, 0])
                .reshape(REP_M, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K_GROUP_SZ)
            )

            # Load one 128xK scale block for B:
            # b_scale_desc.block_shape = [1, 1, rep_k, 2, 256]
            scale_b_block = b_scale_desc.load([pid_b, 0, offs_scale_k, 0, 0])
            # shape: [1, 1, rep_k, 2, 256]

            # Reconstruct full 128 rows (N_BLOCK = 128) as in the docs / original tutorial
            # 1 * 1 * rep_k * 2 * 256 elements = rep_k * 512
            scale_b = (
                scale_b_block.reshape(1, REP_K, 32, 4, 4)  # (1, rep_k, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)  # (1, 4, 32, rep_k, 4)
                .reshape(128, BLOCK_K_GROUP_SZ)  # (128, rep_k * 4) = (128, K/GROUP_SZ)
            )

            # Problem: Scale factors are strided s.t. [128, ...]. This means we
            # need block sizes of _at least_ 128 else change the scale tensor.
            # In our GEMV case, we are only using one column of N, so we'd like
            # BLOCK_N = 1. But this doesn't play nice with tensor cores. Our N
            # is actually padded to 128 to match the stride.
            #
            # Now we'd prefer not to load a block of 128 to only use one element,
            # since we're memory-bound as it is. We'd also like not to break our
            # special nvidia block tiling. Maybe there is a more elegant way of doing
            # this with cute layouts, but here's a simple solution:
            #
            # We alway load the full scale_b block [128, ...], but depcouple this
            # from the BLOCK_N. As long as our BLOCK_N < 128, consumer will only load
            # the first BLOCK_N elems of data, and the extra loaded sizes will not be
            # touched.
            #
            # This allows us to use a BLOCK_N as small as 32 (smallest TC). We do do
            # a redundant load of an extra 96 elems of scale_b, but this ~tiny in
            # comparison with the size of the B.
            accumulator = tl.dot_scaled(
                a,  # [M, K/2]
                scale_a,  # [M, K/GROUP_SZ]
                "e2m1",
                b.T,  # [K/2, N]
                scale_b,  # [128, K/GROUP_SZ]
                "e2m1",
                accumulator,
            )

        # C: [M, L]; store only [M, 1] (masked store)
        c_off = (
            (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_cm  #
            + (pid_b + tl.arange(0, BLOCK_N))[None, :] * stride_cl
        )
        c_mask = (
            (offs_am + tl.arange(0, BLOCK_M) < M)[:, None]  #
            & (tl.arange(0, BLOCK_N) < 1)[None, :]  # Only store the first col
        )
        tl.store(c_ptr + c_off, accumulator.to(output_dtype), mask=c_mask)


def custom_kernel(data):
    a_tensor, b_tensor, _, _, sfa_permuted, sfb_permuted, c_tensor = data
    BLOCK_M = 128
    BLOCK_N = 32  # Lowest for TC usage
    BLOCK_K = 512
    GROUP_SZ = 16  # NVFP4

    REP_M = BLOCK_M // 128
    REP_K = BLOCK_K // GROUP_SZ // 4

    SM_MULT = 1

    M, K_half, L = a_tensor.shape
    N = b_tensor.shape[0]
    assert N == 128, "Expected b_tensor shape [128, K/2, L] (N padded to 128)"

    K = 2 * K_half

    ELEM_PER_BYTE = 2  # for nvfp4

    # Triton doesn't like float8 containers, and we want K as last dim.
    a_tma = a_tensor.view(torch.uint8).permute(0, 2, 1)  # [M, K/2, L] -> [M, L, K/2]
    b_tma = b_tensor.view(torch.uint8).permute(0, 2, 1)  # [N, K/2, L] -> [N, L, K/2]

    a_desc = TensorDescriptor.from_tensor(
        a_tma,
        block_shape=[BLOCK_M, 1, BLOCK_K // ELEM_PER_BYTE],
    )
    b_desc = TensorDescriptor.from_tensor(
        b_tma,
        block_shape=[BLOCK_N, 1, BLOCK_K // ELEM_PER_BYTE],
    )

    # Scales: invert CuTe layout and pack for TMA
    # sfa_permuted: [32, 4, rest_m, 4, rest_k, L]
    # sfb_permuted: [32, 4, rest_n, 4, rest_k, L]
    rest_m = M // 128
    rest_n = N // 128  # = 1
    rest_k = triton.cdiv(K, GROUP_SZ) // 4

    # Permute to [L, rest_m, rest_k, 32, 4, 4]
    sfa_back = sfa_permuted.permute(5, 2, 4, 0, 1, 3)
    sfb_back = sfb_permuted.permute(5, 2, 4, 0, 1, 3)
    assert sfa_back.shape == (L, rest_m, rest_k, 32, 4, 4)
    assert sfb_back.shape == (L, rest_n, rest_k, 32, 4, 4)

    # Pack final three dims: (L, rest_m, rest_k, 32, 4, 4) -> (L, rest_m, rest_k, 2, 256)
    a_scale_packed = sfa_back.view(L, rest_m, rest_k, 2, 256)
    b_scale_packed = sfb_back.view(L, rest_n, rest_k, 2, 256)

    a_scale_desc = TensorDescriptor.from_tensor(
        a_scale_packed,
        block_shape=[1, REP_M, REP_K, 2, 256],
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        b_scale_packed,
        block_shape=[1, 1, REP_K, 2, 256],  # [:, rep_n (1), ...]
    )

    stride_cm, _, stride_cl = c_tensor.stride()  # (M, 1, L)

    # Persistent grid
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    total_tiles = num_tiles_m * L
    num_sms = torch.cuda.get_device_properties(a_tensor.device).multi_processor_count
    num_programs = min(total_tiles, num_sms * SM_MULT)
    grid = (num_programs,)

    kernel = block_scaled_batched_gemv_kernel[grid](  # brrr
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_tensor,
        stride_cm,
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
        REP_K,
    )

    return c_tensor
