#pragma once

#include "kernels/1_naive.hpp"
#include "kernels/2_kernel_global_mem_coalesce.hpp"
#include "kernels/3_kernel_shared_mem_blocking.hpp"
#include "kernels/4_kernel_1D_blocktiling.hpp"
#include "kernels/5_kernel_2D_blocktiling.hpp"
#include "kernels/6_kernel_vectorize.hpp"
#include "kernels/7_kernel_resolve_bank_conflicts.hpp"
#include "kernels/8_kernel_bank_extra_col.hpp"
#include "kernels/9_kernel_autotuned.hpp"
#include "kernels/10_kernel_warptiling.hpp"
// #include "kernels/11_kernel_double_buffering.hpp"
// #include "kernels/12_kernel_double_buffering.hpp"
