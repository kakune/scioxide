//! dense family: numerical, resource and transaction contracts.

mod support;

#[path = "families/dense_blas_forms.rs"]
mod dense_blas_forms;
#[path = "families/dense_kernel_contracts.rs"]
mod dense_kernel_contracts;
#[path = "families/gemm_workspace.rs"]
mod gemm_workspace;
#[path = "families/level2_kernels.rs"]
mod level2_kernels;
#[path = "families/matrix_method_resources.rs"]
mod matrix_method_resources;

#[path = "families/dense_workspace_contracts.rs"]
mod dense_workspace_contracts;
