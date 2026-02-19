from torch.utils.cpp_extension import load_inline
import torch

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Kernel configuration for NVFP4 block-scaled GEMM with FP16 output

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutDTag = LayoutCTag;
constexpr int AlignmentD = AlignmentC;

using ElementSFA = cutlass::float_ue4m3_t;
using ElementSFB = cutlass::float_ue4m3_t;

using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = Shape<_128, _64, _256>;
using ClusterShape = Shape<_1, _2, _1>;

using PerSmTileShapeMNK = Shape<_128, _64, _256>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    PerSmTileShapeMNK,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
    // tbd: , fusionoperation (e.g. generate scale factor outputs)
>::CollectiveOp;


using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<8>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100
>::CollectiveOp;


using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));

torch::Tensor run_gemm(std::vector<torch::Tensor> data) {

    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;;


    auto a = data[1];  // [M, K/2] uint8 packed FP4
    auto b = data[0];  // [N, K/2] uint8 packed FP4
    auto scale_a = data[5];  // Permuted FP8 scales
    auto scale_b = data[4];  // Permuted FP8 scales
    //auto c = data[6];  // [M, N] FP16 output


    int m = a.size(0);
    int k = a.size(1) * 2;  // Unpacked K dimension
    int n = b.size(0);

    auto c = torch::empty({m, n, 1}, torch::dtype(torch::kFloat16).device(a.device()));

    auto a_ptr = static_cast<typename Gemm::ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<typename Gemm::ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementD*>(c.data_ptr());
    auto sfa_ptr = static_cast<ElementSFA const*>(scale_a.data_ptr());
    auto sfb_ptr = static_cast<ElementSFB const*>(scale_b.data_ptr());


    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    //auto layout_A = cute::make_layout(make_shape(m, k, 1), stride_A);
    //auto layout_B = cute::make_layout(make_shape(n, k, 1), stride_B);
    //auto layout_D = cute::make_layout(make_shape(m, n, 1), stride_D)

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1)
    );
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1)
    );

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        { // Mainloop
            a_ptr, stride_A,
            b_ptr, stride_B,
            sfa_ptr, layout_SFA,
            sfb_ptr, layout_SFB
        },
        { // Epilogue
            {}, // epilogue.thread
            c_ptr, stride_D,
            c_ptr, stride_D
        }
    };

    // TODO: alpha?
    Gemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(arguments);

        // Allocate workspace if needed
    void* workspace_ptr = nullptr;
    torch::Tensor workspace;
    if (workspace_size > 0) {
        workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                  torch::dtype(torch::kUInt8).device(a.device()));
        workspace_ptr = workspace.data_ptr();
    }

    // Initialize
    auto status = gemm_op.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS initialization failed");
    }

    // Run
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS execution failed");
    }

    return c.transpose(0, 1);
}

"""

cpp_source = """
torch::Tensor run_gemm(std::vector<torch::Tensor> data);
"""

module = load_inline(
    name='nvfp4_gemm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['run_gemm'],
    with_cuda=True,
    extra_cflags=[
        '-O3',
        '-std=c++20',
        '-march=native',
    ],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '--extra-device-vectorization',
        '--maxrregcount=128',
        '-arch=sm_100a',
        '-Xptxas=-v',
        '-lineinfo',
        '-std=c++20',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
    ],
    verbose=True,
)

custom_kernel = torch.inference_mode(module.run_gemm)
