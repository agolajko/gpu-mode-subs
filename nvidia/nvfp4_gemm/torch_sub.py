import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked_batched(input_matrix):
    """
    Convert scale factor tensor to blocked format for all batches at once.
    
    Input: [rows, cols, L] where rows could be M or N, cols is K//16
    Output: [flattened_blocked_size, L] where we can index [:, l_idx] for each batch
    """
    rows, cols, l = input_matrix.shape

    # Ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Add batch dimension to all the reshaping operations
    # [n_row_blocks, 128, n_col_blocks, 4, L]
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4, l)
    
    # [n_row_blocks, n_col_blocks, 128, 4, L]
    blocks = blocks.permute(0, 2, 1, 3, 4)
    
    # [n_row_blocks * n_col_blocks, 4, 32, 4, L]
    rearranged = blocks.reshape(-1, 4, 32, 4, l)
    
    # [n_row_blocks * n_col_blocks, 32, 4, 4, L]
    rearranged = rearranged.transpose(1, 2)
    
    # [n_row_blocks * n_col_blocks, 32, 16, L]
    rearranged = rearranged.reshape(-1, 32, 16, l)
    
    # [n_row_blocks * n_col_blocks * 32 * 16, L]
    # This way we can index with [:, l_idx] to get the flattened blocked scales for batch l_idx
    return rearranged.reshape(-1, l)

@torch.compile(options={
    "max_autotune_gemm_backends": "TRITON,ATEN",
    "coordinate_descent_tuning": True,  # More thorough search
    "coordinate_descent_search_radius": 2,
    "triton.cudagraphs": False,
})
# @torch.compile(mode="max-autotune")
def custom_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    scale_a_blocked = to_blocked_batched(sfa_ref_cpu)  # [flattened_size, L]
    scale_b_blocked = to_blocked_batched(sfb_ref_cpu)  # [flattened_size, L]
    
    # Move to GPU once
    scale_a_blocked = scale_a_blocked.cuda()
    scale_b_blocked = scale_b_blocked.cuda()
    results = []

    # Call torch._scaled_mm to compute the GEMM result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = scale_a_blocked[:, l_idx]
        scale_b = scale_b_blocked[:, l_idx]
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, :, l_idx] = res
        results.append(res)
        output = torch.stack(results, dim=2)


    return output

