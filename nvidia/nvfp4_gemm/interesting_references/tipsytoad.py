"""
NVFP4 GEMM
===

Tutorial reference: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

# SCALE FACTORS

For efficient block-scaled matmul, we need to structure our scale factors in the
following format:
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x

However Triton's `tl.dot_scaled` api doesn't expose this format, instead just using
the standard [M, K_groups]. Where it can index simply by:

```python
group_sz = 16
for i in range(m):
  for j in range(k):
      s = scale[i, j // group_sz]
      a = A[i, j] * s
```

In order to get satisfactory perf, our underlying scale tensors should be in
Nvidia's format for hardware, while exposing them in the above format for Triton.

NVIDIA Block layout:
- shape:    [32, 4, rest_m, 4, rest_k, L]
- dims:     [g_m, i_m, tile_m, i_k, tile_k, l]
- M:        g_m * i_m * rest_m
- K_groups: i_k * rest_k

This is a factorization of the "simple" 2D scale grid like so:
- i = rest_m * 128 + i_m * 32 + g_m
- j = rest_k * 4 + i_k

And we can simply map to the original format:
> g_m i_m tile_m i_k tile_k l -> (tile_m i_m g_m) (tile_k i_k) l

## Layout semantics

However there's a problem with this approach. Our tensor is laid out in memory like:

> l tile_m tile_k g_m i_m i_k

Doing this transform in pytorch isn't possible without a copy (which is the whole point,
we want to keep it in this format for hardware).

> l tile_m tile_k g_m i_m i_k -> (tile_m i_m g_m) (tile_k i_k) l
> Not trivially possible! Use reshape(...) instead of view(...) error.

(This may be possible using hierarchical cute layouts, but alas).

To reiterate: The whole point is that we want to **keep the scales in block-scaled
format** for hardware acceleration, but pass the data into `tl.dot_sclaled` as a
block shape [BLOCK_M, SCALE_BLOCK_K].

### Triton Reshape

Triton reshape semantics are fundamentally different to Triton. `torch.reshape` operates
on memory with stride layouts. `tl.reshape` only reindexes the loaded block of memory,
so is (almost) free.

Note: Triton has both `view` and `reshape`. `view` may reorder elements for better
codegen; `tl.reshape(..., can_reorder=False)` (default) guarantees element order is preserved.
We need the latter: block-scaled interleaving must stay intact for hardware.

## TMA size

Further optimzing the TMA load:

Therefore we load the 6D tensor into registers, reshape to [BLOCK_M, BLOCK_K_SCALE]
for `tl.dot_scaled`, while keeping data in hardware-optimal format.


Therefore we'll load the 6D tensor into the block, reshaping from there. Triton has


# Use 5D TMA descriptor [1, rep_m, rep_k, 2, 256]
# With 256 elements we better utilize the L2 and don't require the TMA
# engine to emit many small messages (16B) messages as with 32x16xu8.

"""

import functools
import itertools
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


class Shape(NamedTuple):
    L: int
    M: int
    N: int
    K: int


class Autotune(NamedTuple):
    num_outer_stages: int | None
    num_inner_stages: int | None
    warp_specialize_outer: bool
    warp_specialize_inner: bool
    flatten_outer: bool
    flatten_inner: bool
    swizzle_group_sz: int


def _matmul_launch_metadata(grid, kernel, args):
    M, N, K = args["M"], args["N"], args["K"]
    return {
        "name": f"{kernel.name} [M={M}, N={N}, K={K}]",
        "flops": 2.0 * M * N * K,
    }


def get_heuristics():
    def inner(nargs, dim):
        BLOCK_X = nargs[f"BLOCK_{dim}"]
        # How many "tiles" of tile_m, tile_n, tile_k we need per block
        # Reminder: m: (32 4 rest_m) k: (4 rest_k) l: (l)
        if dim in ("M", "N"):
            REP_X = triton.cdiv(BLOCK_X, 128)  # 32 * 4
        else:  # K
            SCALE_GROUP_SZ = nargs["SCALE_GROUP_SZ"]
            REP_X = BLOCK_X // SCALE_GROUP_SZ // 4  # 4

        assert REP_X > 0
        return REP_X

    return {f"REP_{x}": functools.partial(inner, dim=x) for x in ("M", "N", "K")}


def gemm_prehook(nargs):
    """Update TMA descriptor block shapes based on autotuned parameters."""
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    # weird thing where if I put heuristics first, it won't have block_*, if I
    # put autotune first, prehook will be called before heuristics. So I have
    # to call heuristics manually here.
    reps = {k: f(nargs) for k, f in get_heuristics().items()}
    REP_M = reps["REP_M"]
    REP_N = reps["REP_N"]
    REP_K = reps["REP_K"]
    ELEM_PER_BYTE = nargs["ELEM_PER_BYTE"]

    # We pack two nvfp4 vals for each element of a & b (uint8). Without `// ELEM_PER_BYTE`
    # we would load double the number of elements.
    # - Note: M & N are not "packed" because they are outer dims.
    nargs["a_desc"].block_shape = [1, BLOCK_M, BLOCK_K // ELEM_PER_BYTE]
    nargs["b_desc"].block_shape = [1, BLOCK_N, BLOCK_K // ELEM_PER_BYTE]
    nargs["c_desc"].block_shape = [1, BLOCK_M, BLOCK_N]
    nargs["sfa_desc"].block_shape = [1, REP_M, REP_K, 2, 256]
    nargs["sfb_desc"].block_shape = [1, REP_N, REP_K, 2, 256]


def get_config():
    # fmt: off
    # Block sizes
    block_m = [128]
    block_n = [128]
    block_k = [256, 512]

    # Autotune parameters
    num_outer_stages = [None, 2, 4]
    num_inner_stages = [None, 2, 4]
    warp_specialize_outer = [False, True]
    warp_specialize_inner = [False, True]
    flatten_outer = [False]
    flatten_inner = [False]
    swizzle_group_sz = [4]

    # Kernel launch params
    num_warps = [4]
    num_stages = [2, 3, 4]

    configs = []
    for (
        bm, bn, bk,
        n_outer, n_inner,
        ws_outer, ws_inner,
        flat_outer, flat_inner,
        swizzle,
        nwarps, nstages
    ) in itertools.product(
        block_m, block_n, block_k,
        num_outer_stages, num_inner_stages,
        warp_specialize_outer, warp_specialize_inner,
        flatten_outer, flatten_inner,
        swizzle_group_sz,
        num_warps, num_stages
    ):
        configs.append(
            triton.Config(
                dict(
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    autotune=Autotune(
                        num_outer_stages=n_outer,
                        num_inner_stages=n_inner,
                        warp_specialize_outer=ws_outer, # type: ignore
                        warp_specialize_inner=ws_inner, # type: ignore
                        flatten_outer=flat_outer, # type: ignore
                        flatten_inner=flat_inner, # type: ignore
                        swizzle_group_sz=swizzle,
                    ),
                ),
                num_warps=nwarps,
                num_stages=nstages,
                num_ctas=1,
                pre_hook=gemm_prehook,
            )
        )
    return configs
    # fmt: on


def _config(config: triton.Config):
    class inner:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, s):
            def run(*args, **kwargs):
                nargs = dict(zip(self.fn.arg_names, args))
                nargs = {**nargs, **kwargs, **config.all_kwargs()}
                if config.pre_hook:
                    config.pre_hook(nargs)
                return self.fn[s](**nargs)

            return run

    return inner


"""
L=1 M=128 N=7168 K=16384
38.9440 ms (p20=38.7840, p80=39.0080): BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 256, autotune: Autotune(
    num_outer_stages=2, num_inner_stages=4, warp_specialize_outer=False, warp_specialize_inner=True,
    flatten_outer=False, flatten_inner=False, swizzle_group_sz=4),
num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None

38.9440 ms (p20=38.8160, p80=39.0080): BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 256, autotune: Autotune(num_outer_stages=None, num_inner_stages=None, warp_specialize_outer=True, warp_specialize_inner=True, flatten_outer=False, flatten_inner=False, swizzle_group_sz=4), num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None
38.9440 ms (p20=38.8160, p80=39.0080): BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 256, autotune: Autotune(num_outer_stages=None, num_inner_stages=4, warp_specialize_outer=False, warp_specialize_inner=True, flatten_outer=False, flatten_inner=False, swizzle_group_sz=4), num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
38.9440 ms (p20=38.8160, p80=39.0080): BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 256, autotune: Autotune(num_outer_stages=None, num_inner_stages=4, warp_specialize_outer=False, warp_specialize_inner=True, flatten_outer=False, flatten_inner=False, swizzle_group_sz=4), num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
38.9440 ms (p20=38.8160, p80=39.0080): BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 256, autotune: Autotune(num_outer_stages=None, num_inner_stages=4, warp_specialize_outer=False, warp_specialize_inner=True, flatten_outer=False, flatten_inner=False, swizzle_group_sz=4), num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None




"""


# @triton.autotune(configs=get_config(), key="shape")
@_config(
    triton.Config(
        dict(
            # block
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=256,
            # param
            autotune=Autotune(
                num_outer_stages=2,
                num_inner_stages=4,
                warp_specialize_outer=False,
                warp_specialize_inner=True,
                flatten_outer=False,
                flatten_inner=False,
                swizzle_group_sz=4,
            ),
        ),
        # kernel launch
        num_warps=4,
        num_stages=2,
        num_ctas=1,  # this doesn't play nice
        pre_hook=gemm_prehook,
    )
)
@triton.heuristics(values=get_heuristics())
@triton.jit(launch_metadata=_matmul_launch_metadata)
def persistent_block_scaled_gemm(
    a_desc,
    b_desc,
    c_desc,
    sfa_desc,
    sfb_desc,
    shape: Shape,
    autotune: tl.constexpr,
    out_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REP_M: tl.constexpr,
    REP_N: tl.constexpr,
    REP_K: tl.constexpr,
    ELEM_PER_BYTE: tl.constexpr,
    SCALE_GROUP_SZ: tl.constexpr,
):
    BLOCK_K_ELEM_PER_BYTE: tl.constexpr = BLOCK_K // ELEM_PER_BYTE
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_GROUP_SZ
    pid_sched = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_pid_m, num_pid_n, num_pid_l = (
        tl.cdiv(shape.M, BLOCK_M),
        tl.cdiv(shape.N, BLOCK_N),
        shape.L,
    )
    num_tiles = num_pid_m * num_pid_n * num_pid_l
    for linear in tl.range(
        pid_sched,
        num_tiles,
        num_programs,
        num_stages=autotune.num_outer_stages,
        flatten=autotune.flatten_outer,
        warp_specialize=autotune.warp_specialize_outer,
    ):
        # linear order
        pid_m = linear % num_pid_m
        pid_n = (linear // num_pid_m) % num_pid_n
        pid_l = linear // (num_pid_m * num_pid_n)
        # fmt: off
        # tiled swizzle order
        pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, autotune.swizzle_group_sz)
        # fmt: on
        off_m, off_n, off_l = pid_m * BLOCK_M, pid_n * BLOCK_N, pid_l
        off_m_scale, off_n_scale = pid_m * REP_M, pid_n * REP_N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
        for i in tl.range(
            0,
            tl.cdiv(shape.K, BLOCK_K),
            num_stages=autotune.num_inner_stages,
            flatten=autotune.flatten_inner,
            warp_specialize=autotune.warp_specialize_inner,
        ):
            # a,b: As said before our a & b are packed 2:1 in uint8, so to find
            # our "byte offset", we need to divide by ELEM_PER_BYTE (2).
            #
            # scales: a group of 16 elements of a&b we need a single scale so we
            # Therefore we need a block_shape of [..., BLOCK_K // SCALE_GROUP_SZ]
            off_k_ab = i * BLOCK_K_ELEM_PER_BYTE
            off_k_scale = i * REP_K

            # fmt: off
            a = a_desc.load([off_l, off_m, off_k_ab]).reshape(BLOCK_M, BLOCK_K_ELEM_PER_BYTE)
            b = b_desc.load([off_l, off_n, off_k_ab]).reshape(BLOCK_N, BLOCK_K_ELEM_PER_BYTE)

            sfa = (
                sfa_desc
                .load([off_l, off_m_scale, off_k_scale, 0, 0])  # l tile_m tile_k 2 256
                .reshape([REP_M, REP_K, 32, 4, 4])  # tile_m tile_k g_m [32] i_m [4] i_k [4]
                .trans(0, 3, 2, 1, 4) #  tile_m i_m g_m tile_k i_k
                .reshape(BLOCK_M, BLOCK_K_SCALE, can_reorder=False) # (tile_m i_m g_m) (tile_k i_k)
            )
            sfb = (
                sfb_desc
                .load([off_l, off_n_scale, off_k_scale, 0, 0])  # l tile_n tile_k 2 256
                .reshape([REP_N, REP_K, 32, 4, 4])  # tile_n tile_k g_m [32] i_n [4] i_k [4]
                .trans(0, 3, 2, 1, 4) #  tile_n i_n g_n tile_k i_k
                .reshape(BLOCK_N, BLOCK_K_SCALE, can_reorder=False) # (tile_n i_n g_n) (tile_k i_k)
            )
            # fmt: on

            acc = tl.dot_scaled(a, sfa, "e2m1", b.T, sfb, "e2m1", acc)
        c_desc.store([off_l, off_m, off_n], tl.expand_dims(acc, 0).to(out_dtype))


def custom_kernel(data):
    """
    Triton implementation of block-scale nvfp4 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float4_e2m1fn_x2] of shape [m, k, l],
            b: torch.Tensor[float4_e2m1fn_x2] of shape [n, k, l],
            sfa: torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l],
            sfb: torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l],
            sfa_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_m, 4, rest_k, l],
            sfb_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_n, 4, rest_k, l],
            c: torch.Tensor[float16] of shape [m, n, l]
    Returns:
        Tensor containing output in float16
        c: torch.Tensor[float16] of shape [m, n, l]
    """
    a, b, _, _, sfa, sfb, c = data
    M, K_half, L = a.shape
    N, _, _ = b.shape
    assert L == 1, f"input is batched {a.shape=}"

    SCALE_GROUP_SZ = 16
    ELEM_PER_BYTE = 2
    out_dtype = tl.float16
    acc_dtype = tl.float32
    # float4_e2m1fn_x2: x2 packed float4, so torch reports shape as if its a uint8
    K = K_half * 2

    # mkl->lmk; the tensor is actually allocated this way
    # unpack float4 containers, don't play nice with Triton
    a, b, c = (
        a.permute(2, 0, 1).view(torch.uint8),
        b.permute(2, 0, 1).view(torch.uint8),
        c.permute(2, 0, 1),  # still fp16
    )
    assert a.is_contiguous() and b.is_contiguous(), f"a&b not contiguous {a.stride()=}"

    # [g_m, i_m, tile_m, i_k, tile_k, l]
    # => m: (32 4 rest_m) k: (4 rest_k) l: (l)
    # => (M) (K // SCALE_GROUP_SZ) (L)
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % SCALE_GROUP_SZ == 0 and (K // SCALE_GROUP_SZ) % 4 == 0
    assert K % 256 == 0
    tile_m = M // 128
    tile_n = N // 128
    tile_k = K // SCALE_GROUP_SZ // 4
    assert sfa.shape == (32, 4, tile_m, 4, tile_k, L), (
        f"expected != actual: {sfa.shape=} != (32, 4, {tile_m=}, 4, {tile_k=}, {L=})"
    )

    # TMA doesn't support 6D tensors,
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    # (See block scaling tutorial)
    # g_m [32] i_m [4] tile_m i_k [4] tile_k l
    # -> tile_m tile_k (g_m i_m i_k) [512]
    # -> tile_m tile_k (2 256)

    # g_m i_m tile_m i_k tile_k l
    # -> l tile_m tile_k g_m [32] i_m [4] i_k [4]
    # -> l tile_m tile_k (2, 256)
    # From tutorial:
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    sfa = sfa.permute(5, 2, 4, 0, 1, 3).view(L, tile_m, tile_k, 2, 256)
    sfb = sfb.permute(5, 2, 4, 0, 1, 3).view(L, tile_n, tile_k, 2, 256)
    assert sfa.is_contiguous() and sfb.is_contiguous(), (
        f"sfa & sfb should be contigous: {sfa.stride()=} {sfb.stride()=}"
    )

    # dummy block shape. pre_hook will set actual values
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[1, 1, 1])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[1, 1, 1])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[1, 1, 1])
    sfa_desc = TensorDescriptor.from_tensor(sfa, block_shape=[1, 1, 1, 1, 1])
    sfb_desc = TensorDescriptor.from_tensor(sfb, block_shape=[1, 1, 1, 1, 1])

    # launch
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (num_sms,)
    persistent_block_scaled_gemm[grid](  # type: ignore go brrrrrrr
        a_desc,
        b_desc,
        c_desc,
        sfa_desc,
        sfb_desc,
        shape=Shape(L, M, N, K),
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        ELEM_PER_BYTE=ELEM_PER_BYTE,
        SCALE_GROUP_SZ=SCALE_GROUP_SZ,
    )

    # if Shape(L, M, N, K) not in [(1, 128, 7168, 16384), (1, 128, 4096, 7168)]:
    #     timings = persistent_block_scaled_gemm.configs_timings
    #     top5 = sorted(timings.items(), key=lambda x: x[1])[:5]
    #     print(f"{L=} {M=} {N=} {K=}")
    #     for cfg, t in top5:
    #         print(
    #             f"{t[0] * 1000:.8f} ms (p20={t[1] * 1000:.8f}, p80={t[2] * 1000:.8f}): {cfg}\n"
    #         )
    #     assert False

    return c.view(M, N, L)  # unsqueeze
