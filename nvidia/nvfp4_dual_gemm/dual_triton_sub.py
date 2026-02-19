import functools

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _dualmm_launch_metadata(grid, kernel, args):
    M, N, K = args["M"], args["N"], args["K"]
    # 2 GEMMs => ~4*M*N*K FLOPs (ignoring epilogue)
    return {
        "name": f"{kernel.name} [dualmm] [M={M}, N={N}, K={K}]",
        "flops": 4.0 * M * N * K,
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
@triton.jit(launch_metadata=_dualmm_launch_metadata)
def block_scaled_batched_dualmm_kernel(
    a_desc,
    a_scale_desc,
    b1_desc,
    b1_scale_desc,
    b2_desc,
    b2_scale_desc,
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
        pid_m = linear % num_pid_m
        pid_n = (linear // num_pid_m) % num_pid_n
        pid_b = linear // (num_pid_m * num_pid_n)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_scale_m = pid_m * REP_M
        offs_scale_n = pid_n * REP_N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        for i in tl.range(
            0,
            tl.cdiv(K, BLOCK_K),
            num_stages=NUM_INNER_STAGES,
            warp_specialize=WARP_SPECIALIZE_INNER,
        ):
            offs_k = i * BLOCK_K_ELEM_PER_BYTE
            offs_scale_k = i * REP_K

            # Load A
            a = a_desc.load([offs_am, pid_b, offs_k]).reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)

            # Load B1
            b1 = b1_desc.load([offs_bn, pid_b, offs_k]).reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            # Load scale_a
            scale_a = (
                a_scale_desc.load([pid_b, offs_scale_m, offs_scale_k, 0, 0])
                .reshape(REP_M, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K_GROUP_SZ)
            )

            # Load scale_b1
            scale_b1 = (
                b1_scale_desc.load([pid_b, offs_scale_n, offs_scale_k, 0, 0])
                .reshape(REP_N, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_N, BLOCK_K_GROUP_SZ)
            )

            acc_gate = tl.dot_scaled(a, scale_a, "e2m1", b1.T, scale_b1, "e2m1", acc_gate)

        # -------------------------
        # Loop 2: up = A @ B2
        # -------------------------
        for i in tl.range(
            0,
            tl.cdiv(K, BLOCK_K),
            num_stages=NUM_INNER_STAGES,
            warp_specialize=WARP_SPECIALIZE_INNER,
        ):
            offs_k = i * BLOCK_K_ELEM_PER_BYTE
            offs_scale_k = i * REP_K

            # Load A again
            a = a_desc.load([offs_am, pid_b, offs_k]).reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)

            # Load B2
            b2 = b2_desc.load([offs_bn, pid_b, offs_k]).reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            # Load scale_a again
            scale_a = (
                a_scale_desc.load([pid_b, offs_scale_m, offs_scale_k, 0, 0])
                .reshape(REP_M, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_M, BLOCK_K_GROUP_SZ)
            )

            # Load scale_b2
            scale_b2 = (
                b2_scale_desc.load([pid_b, offs_scale_n, offs_scale_k, 0, 0])
                .reshape(REP_N, REP_K, 32, 4, 4)
                .trans(0, 3, 2, 1, 4)
                .reshape(BLOCK_N, BLOCK_K_GROUP_SZ)
            )

            acc_up = tl.dot_scaled(a, scale_a, "e2m1", b2.T, scale_b2, "e2m1", acc_up)

        # Epilogue
        gate = acc_gate * tl.sigmoid(acc_gate)   # SiLU
        out = gate * acc_up

        c_off = (
            (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_cm
            + (offs_bn + tl.arange(0, BLOCK_N))[None, :] * stride_cn
            + pid_b * stride_cl
        )
        c_mask = (
            (offs_am + tl.arange(0, BLOCK_M) < M)[:, None]
            & (offs_bn + tl.arange(0, BLOCK_N) < N)[None, :]
        )
        tl.store(c_ptr + c_off, out.to(output_dtype), mask=c_mask)


def custom_kernel(data):
    """
    Expected `data` tuple layout (mirroring your existing one):
      a_tensor, b1_tensor, b2_tensor, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c_tensor
    Adjust indexing to your real packing if needed.
    """
    a_tensor, b1_tensor, b2_tensor, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c_tensor = data


    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    GROUP_SZ = 16

    REP_M = BLOCK_M // 128
    REP_N = BLOCK_N // 128
    REP_K = BLOCK_K // GROUP_SZ // 4

    SM_MULT = 2

    M, K_half, L = a_tensor.shape
    N, _, _ = b1_tensor.shape
    # assume b2_tensor.shape matches b1_tensor.shape
    K = 2 * K_half
    ELEM_PER_BYTE = 2

    # TMA descriptors (uint8 view + permute like your original)
    a_tma = a_tensor.view(torch.uint8).permute(0, 2, 1)   # [M, L, K/2]
    b1_tma = b1_tensor.view(torch.uint8).permute(0, 2, 1) # [N, L, K/2]
    b2_tma = b2_tensor.view(torch.uint8).permute(0, 2, 1) # [N, L, K/2]

    a_desc = TensorDescriptor.from_tensor(a_tma, block_shape=[BLOCK_M, 1, BLOCK_K // ELEM_PER_BYTE])
    b1_desc = TensorDescriptor.from_tensor(b1_tma, block_shape=[BLOCK_N, 1, BLOCK_K // ELEM_PER_BYTE])
    b2_desc = TensorDescriptor.from_tensor(b2_tma, block_shape=[BLOCK_N, 1, BLOCK_K // ELEM_PER_BYTE])

    # Scale descriptors
    rest_m = M // 128
    rest_n = N // 128
    rest_k = triton.cdiv(K, GROUP_SZ) // 4

    sfa_back = sfa_permuted.permute(5, 2, 4, 0, 1, 3)
    sfb1_back = sfb1_permuted.permute(5, 2, 4, 0, 1, 3)
    sfb2_back = sfb2_permuted.permute(5, 2, 4, 0, 1, 3)

    a_scale_packed = sfa_back.view(L, rest_m, rest_k, 2, 256)
    b1_scale_packed = sfb1_back.view(L, rest_n, rest_k, 2, 256)
    b2_scale_packed = sfb2_back.view(L, rest_n, rest_k, 2, 256)

    a_scale_desc = TensorDescriptor.from_tensor(a_scale_packed, block_shape=[1, REP_M, REP_K, 2, 256])
    b1_scale_desc = TensorDescriptor.from_tensor(b1_scale_packed, block_shape=[1, REP_N, REP_K, 2, 256])
    b2_scale_desc = TensorDescriptor.from_tensor(b2_scale_packed, block_shape=[1, REP_N, REP_K, 2, 256])

    stride_cm, stride_cn, stride_cl = c_tensor.stride()

    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n * L

    num_sms = torch.cuda.get_device_properties(a_tensor.device).multi_processor_count
    num_programs = min(total_tiles, num_sms * SM_MULT)
    grid = (num_programs,)

    block_scaled_batched_dualmm_kernel[grid](
        a_desc,
        a_scale_desc,
        b1_desc,
        b1_scale_desc,
        b2_desc,
        b2_scale_desc,
        c_tensor,
        stride_cm,
        stride_cn,
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
        REP_N,
        REP_K,
    )

    return c_tensor
