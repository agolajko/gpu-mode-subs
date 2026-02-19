import torch
from torch.utils.cpp_extension import load_inline
import os

input_t = tuple
output_t = torch.Tensor

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::half_t;
using ElementC = void;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 16;
constexpr int AlignmentC = 1;

using ElementAccumulator = float;
using ElementCompute = cutlass::half_t;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = Shape<_128, _64, _256>;
using ClusterShapeMainloop = Shape<_1, _4, _1>;
using ClusterShapeEpilogue = Shape<_1, _2, _1>;

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized1Sm;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShapeEpilogue,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShapeMainloop,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static Gemm g_gemm;

void nvfp4_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor sfa_perm, torch::Tensor sfb_perm,
    torch::Tensor c, int m, int n, int k, int l)
{
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, l});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, l});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, l});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, l},
        {reinterpret_cast<typename ElementA::DataType*>(a.data_ptr()), stride_A,
        reinterpret_cast<typename ElementB::DataType*>(b.data_ptr()), stride_B,
        reinterpret_cast<typename ElementA::ScaleFactorType*>(sfa_perm.data_ptr()), layout_SFA,
        reinterpret_cast<typename ElementB::ScaleFactorType*>(sfb_perm.data_ptr()), layout_SFB},
        {{ElementCompute(1.0f), ElementCompute(0.0f)}, nullptr, {}, reinterpret_cast<ElementD*>(c.data_ptr()), stride_D}
    };

    g_gemm.initialize(args, nullptr);
    g_gemm.run();
}

#else
void nvfp4_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                torch::Tensor, int, int, int, int) {
    TORCH_CHECK(false, "SM100 not supported");
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm, "NVFP4 GEMM");
}
'''

cpp_source = """
#include <torch/extension.h>
void nvfp4_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sfa_perm, 
                torch::Tensor sfb_perm, torch::Tensor c, int m, int n, int k, int l);
"""

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")
cuda_include = os.environ.get("CUDA_INCLUDE_DIR", "/usr/local/cuda/include")

module = load_inline(
    name="nvfp4_gemm_1x4x1",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    extra_include_paths=[f"{cutlass_path}/include", f"{cutlass_path}/tools/util/include", cuda_include],
    extra_cuda_cflags=["-std=c++17", "-arch=sm_100a", "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
                       "-O3", "--use_fast_math", "--ftz=true", "--prec-div=false", "--prec-sqrt=false"],
    verbose=False,
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, sfa_perm, sfb_perm, c = data
    m, n, l = c.shape[0], c.shape[1], c.shape[2]
    k = a.shape[1] * 2
    module.nvfp4_gemm(a, b, sfa_perm, sfb_perm, c, m, n, k, l)
    return c
