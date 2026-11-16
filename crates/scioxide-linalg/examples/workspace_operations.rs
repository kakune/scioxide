//! Caller-owned storage, execution contexts and prepared operations.
//! Run with `cargo run -p scioxide-linalg --example workspace_operations`.
use scioxide_core::SciOxideError;

fn actions_cancelled_norm_estimate() -> Result<(), SciOxideError> {
    use scioxide_core::{CancellationToken, ExecutionContext, ExecutionStopReason};
    use scioxide_linalg::{Matrix, OneNormEstimator, SpectralTermination};
    let a = Matrix::<f64>::identity(2)?;
    let operator = a.view().try_finite()?;
    let method = OneNormEstimator::new();
    let mut workspace = method.workspace(&operator)?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let report = method.estimate_scoped(
        &operator,
        &mut workspace,
        ExecutionContext::sequential_native().with_cancellation(&cancellation),
    )?;
    assert_eq!(
        report.termination(),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    // No product completed before the cancellation was observed.
    assert_eq!(report.estimate(), 0.);
    Ok::<(), scioxide_core::SciOxideError>(())
}

#[cfg(feature = "blas")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples assert successful results for fixed valid inputs"
)]
fn blas_provider_dispatch() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::backend::ScalarKind;
    use scioxide_core::{CapabilityError, Complex32, Complex64};
    use scioxide_linalg::{
        BlasCall, BlasCapabilities, BlasDispatch, BlasFallbackPolicy, BlasIntegerWidth, BlasLayout,
        BlasOperation, BlasProvider, BlasTransform, BlasUlp, LinalgExecutionContext,
    };

    struct MyBlas;

    impl BlasProvider for MyBlas {
        fn name(&self) -> &'static str {
            "my-blas"
        }
        fn capabilities(&self) -> BlasCapabilities {
            BlasCapabilities::empty(BlasIntegerWidth::Lp64)
                .with_layout(BlasLayout::ColumnMajor)
                .with_transform(BlasTransform::None)
                .with_operation(ScalarKind::F64, BlasOperation::Gemm, BlasUlp::new(4))
                .expect("GEMM is defined for f64")
        }
        fn execute_f64(
            &self,
            call: BlasCall<'_, f64>,
            _workers: NonZeroUsize,
        ) -> Result<(), CapabilityError> {
            match call {
                // Pass a, b and c (values, rows, cols, leading_dimension) to the library's DGEMM.
                BlasCall::Gemm { .. } => Ok(()),
                _ => Err(CapabilityError::new("routine not declared")),
            }
        }
        fn execute_f32(
            &self,
            _: BlasCall<'_, f32>,
            _: NonZeroUsize,
        ) -> Result<(), CapabilityError> {
            Err(CapabilityError::new("routine not declared"))
        }
        fn execute_complex32(
            &self,
            _: BlasCall<'_, Complex32>,
            _: NonZeroUsize,
        ) -> Result<(), CapabilityError> {
            Err(CapabilityError::new("routine not declared"))
        }
        fn execute_complex64(
            &self,
            _: BlasCall<'_, Complex64>,
            _: NonZeroUsize,
        ) -> Result<(), CapabilityError> {
            Err(CapabilityError::new("routine not declared"))
        }
        // execute_f32, execute_complex32 and execute_complex64 follow the same pattern.
    }

    let provider = MyBlas;
    let blas = BlasDispatch::new(&provider, BlasFallbackPolicy::Reference);
    let context = LinalgExecutionContext::sequential().with_blas(blas);
    assert!(context.blas().is_some());
    Ok(())
}

fn decompose_householder_trust_region_solve_trust_region_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::*;
    let a = Matrix::identity(2)?;
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )?;
    let mut out = [0.0_f64; 2];
    // The unconstrained solution (3, 4) has norm 5 > radius 1.
    let report = factor.solve_trust_region_into(
        &[3.0, 4.0][..],
        &[1.0, 1.0],
        1.0,
        0.0,
        &DirectSolveOptions::new(RankPolicy::Exact),
        VectorViewMut::from_slice(&mut out),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert!(
        report
            .step_norm()
            .is_some_and(|norm| (norm - 1.0).abs() < 0.1)
    );
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn decompose_lu_inverse_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix};
    let options = LinalgExecutionOptions::new().with_byte_limit(1 << 20);
    let factor = LuDecomposition::factor_with_options(&Matrix::<f64>::identity(2)?, &options)?;
    let mut output = Matrix::zeros(2, 2)?;
    let mut workspace = LinalgWorkspace::new();
    factor.inverse_into(
        &options,
        output.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let condition = factor.condition_estimate_scoped(
        &options,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!((condition.reciprocal().value() - 1.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples assert successful results for fixed valid inputs"
)]
fn decompose_minimum_norm_householder_minimum_norm_plan() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        DirectSolveOptions, HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
        LinalgExecutionOptions, Matrix, RankPolicy,
    };
    let options = LinalgExecutionOptions::new();
    // x₁ + x₂ = 2 has the minimum-norm solution (1, 1).
    let a = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0])?;
    let plan = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )?
    .into_minimum_norm(
        &(DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
            .with_execution_options(options),
    )?;
    let result = plan.solve(&[2.0][..])?;
    assert!((result.solution().unwrap().as_slice()[0] - 1.0).abs() < 1e-14);
    let mut workspace = plan.workspace(1)?;
    let mut x = [0.0; 2];
    let report = plan.solve_into(
        &[2.0][..],
        scioxide_linalg::VectorViewMut::from_slice(&mut x),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.rank().rank(), 1);
    Ok::<(), scioxide_core::SciOxideError>(())
}

#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples assert successful results for fixed valid inputs"
)]
fn eigen_general_decompose_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralEigenDecomposition, GeneralEigenOptions, GeneralEigenWorkspaceRequest, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 0.0, 3.0])?;
    let options = GeneralEigenOptions::new();
    let request = GeneralEigenWorkspaceRequest::new(2, &options);
    let mut workspace = GeneralEigenDecomposition::<f64>::workspace::<f64>(&request)?;
    let mut decomposition = GeneralEigenDecomposition::empty();
    GeneralEigenDecomposition::decompose_into(
        &a,
        &options,
        &mut decomposition,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert!(
        decomposition
            .right_residuals()
            .unwrap()
            .iter()
            .all(|r| *r < 1e-12)
    );
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_general_compute_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralEigenOptions, GeneralEigenWorkspaceRequest, GeneralEigenvaluesBuffer, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 0.0, 3.0])?;
    let options = GeneralEigenOptions::new();
    let request = GeneralEigenWorkspaceRequest::new(2, &options);
    let mut workspace = GeneralEigenvaluesBuffer::<f64>::workspace::<f64>(&request)?;
    let mut values = GeneralEigenvaluesBuffer::empty();
    GeneralEigenvaluesBuffer::compute_into(
        &a,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert!((values.eigenvalues()[1].re - 3.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_generalized_decompose_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralizedHermitianEigenDecomposition as Eigen, GeneralizedHermitianEigenWorkspaceRequest,
        HermitianEigenOptions, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 6.0])?;
    let b = Matrix::<f64>::identity(2)?;
    let options = HermitianEigenOptions::default();
    let mut workspace =
        Eigen::<f64>::workspace(&GeneralizedHermitianEigenWorkspaceRequest::new(2, &options))?;
    let mut eigen = Eigen::empty();
    Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut eigen,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(eigen.residuals().len(), 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_generalized_compute_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralizedHermitianEigenWorkspaceRequest, GeneralizedHermitianEigenvaluesBuffer as Values,
        HermitianEigenOptions, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 6.0])?;
    let b = Matrix::<f64>::identity(2)?;
    let options = HermitianEigenOptions::default();
    let mut workspace = Values::<f64>::workspace::<f64>(
        &GeneralizedHermitianEigenWorkspaceRequest::new(2, &options),
    )?;
    let mut values = Values::empty();
    Values::compute_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(values.eigenvalues(), &[2.0, 6.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_hermitian_tridiagonal_values() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        HermitianEigenOptions, HermitianEigenWorkspaceRequest, HermitianEigenvaluesBuffer,
    };
    let options = HermitianEigenOptions::default();
    let request = HermitianEigenWorkspaceRequest::tridiagonal(3, &options);
    let mut workspace = HermitianEigenvaluesBuffer::<f64>::workspace::<f64>(&request)?;
    let mut eigen = HermitianEigenvaluesBuffer::empty();
    HermitianEigenvaluesBuffer::tridiagonal_into(
        &[1.0, 2.0, 3.0],
        &[0.5, 0.5],
        &options,
        &mut eigen,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(eigen.eigenvalues().len(), 3);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_hermitian_tridiagonal_vectors() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    };
    let options = HermitianEigenOptions::default();
    let request = HermitianEigenWorkspaceRequest::tridiagonal(3, &options);
    let mut workspace = HermitianEigenDecomposition::<f64>::workspace(&request)?;
    let mut eigen = HermitianEigenDecomposition::empty();
    HermitianEigenDecomposition::tridiagonal_into(
        &[1.0, 2.0, 3.0],
        &[0.5, 0.5],
        &options,
        &mut eigen,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(eigen.eigenvalues().len(), 3);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_hermitian_decompose_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0])?;
    let options = HermitianEigenOptions::default();
    let request = HermitianEigenWorkspaceRequest::new(2, &options);
    let mut workspace = HermitianEigenDecomposition::<f64>::workspace(&request)?;
    let mut eigen = HermitianEigenDecomposition::empty();
    HermitianEigenDecomposition::decompose_into(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut eigen,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(eigen.eigenvectors().cols(), 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_hermitian_compute_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        HermitianEigenOptions, HermitianEigenWorkspaceRequest, HermitianEigenvaluesBuffer, Matrix,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0])?;
    let options = HermitianEigenOptions::default();
    let request = HermitianEigenWorkspaceRequest::new(2, &options);
    let mut workspace = HermitianEigenvaluesBuffer::<f64>::workspace::<f64>(&request)?;
    let mut values = HermitianEigenvaluesBuffer::empty();
    HermitianEigenvaluesBuffer::compute_into(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(values.eigenvalues().len(), 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_qz_decompose_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralizedSchurDecomposition, GeneralizedSchurWorkspaceRequest, Matrix, SchurOptions,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 3.0])?;
    let b = Matrix::<f64>::identity(2)?;
    let options = SchurOptions::new();
    let request = GeneralizedSchurWorkspaceRequest::new(2, &options);
    let mut workspace = GeneralizedSchurDecomposition::<f64>::workspace(&request)?;
    let mut qz = GeneralizedSchurDecomposition::empty();
    GeneralizedSchurDecomposition::decompose_into(
        &a,
        &b,
        &options,
        &mut qz,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert!(qz.eigenvalues().iter().all(|value| value.is_finite()));
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_qz_reorder_reorder_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        GeneralizedSchurDecomposition, GeneralizedSchurWorkspaceRequest, Matrix,
        SchurReorderOptions,
    };
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 0.0, 3.0])?;
    let b = Matrix::<f64>::identity(2)?;
    let mut qz = GeneralizedSchurDecomposition::decompose(&a, &b)?;
    let options = SchurReorderOptions::new();
    let request = GeneralizedSchurWorkspaceRequest::reorder(2, &options);
    let mut workspace = GeneralizedSchurDecomposition::<f64>::workspace(&request)?;
    let report = qz.reorder_in_place(
        &[false, true],
        &options,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.swaps(), 1);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_qz_types_generalized_schur_workspace_request() -> Result<(), SciOxideError> {
    use scioxide_core::Capacity;
    use scioxide_linalg::{
        Estimation, GeneralizedSchurDecomposition, GeneralizedSchurWorkspaceRequest, SchurOptions,
        SchurReorderOptions,
    };
    let factor = GeneralizedSchurWorkspaceRequest::new(4, &SchurOptions::<f64>::new());
    let reorder = GeneralizedSchurWorkspaceRequest::reorder(
        4,
        &SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate),
    );
    let requirements = GeneralizedSchurDecomposition::<f64>::workspace_requirements(&factor)?
        .max(&GeneralizedSchurDecomposition::<f64>::workspace_requirements(&reorder)?);
    let mut workspace = GeneralizedSchurDecomposition::<f64>::workspace(&factor)?;
    workspace.try_reserve(&requirements)?;
    assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_schur_decompose_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{Matrix, SchurDecomposition, SchurOptions, SchurWorkspaceRequest};
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 0.0, 3.0])?;
    let options = SchurOptions::new();
    let mut workspace =
        SchurDecomposition::<f64>::workspace(&SchurWorkspaceRequest::new(2, &options))?;
    let mut schur = SchurDecomposition::empty();
    SchurDecomposition::decompose_into(
        &a,
        &options,
        &mut schur,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(schur.eigenvalues().len(), 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn eigen_schur_reorder_reorder_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{Matrix, SchurDecomposition, SchurReorderOptions, SchurWorkspaceRequest};
    let a = Matrix::from_vec(3, 3, vec![1.0_f64, 2.0, 3.0, 0.0, 4.0, 6.0, 0.0, 0.0, 9.0])?;
    let mut schur = SchurDecomposition::decompose(&a)?;
    let options = SchurReorderOptions::new();
    let mut workspace =
        SchurDecomposition::<f64>::workspace(&SchurWorkspaceRequest::reorder(3, &options))?;
    let report = schur.reorder_in_place(
        &[false, false, true],
        &options,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.selected_dimension(), 1);
    assert!((schur.eigenvalues()[0].re - 9.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn equations_riccati_solve_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        DirectSolveTermination, Matrix, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem,
    };
    let one = Matrix::<f64>::identity(1)?;
    let zero = Matrix::<f64>::zeros(1, 1)?;
    let problem = RiccatiProblem {
        a: zero.view(),
        b: one.view(),
        q: one.view(),
        r: one.view(),
        e: None,
        s: None,
    };
    let plan = RiccatiPlan::new(problem, RiccatiKind::Continuous, RiccatiOptions::new())?;
    let mut workspace = plan.workspace(())?;
    let (mut x, mut k) = (Matrix::zeros(1, 1)?, Matrix::zeros(1, 1)?);
    let report = plan.solve_into(
        x.view_mut(),
        k.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    assert!(report.work() > 0);
    // X = 1 solves 1 − x² = 0, so K = R⁻¹ Bᴴ X = 1.
    assert!((k.view().get(0, 0)? - 1.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn equations_riccati_build() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{Matrix, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem};
    let one = Matrix::<f64>::identity(1)?;
    let half = Matrix::from_vec(1, 1, vec![0.5_f64])?;
    let problem = RiccatiProblem {
        a: half.view(),
        b: one.view(),
        q: one.view(),
        r: one.view(),
        e: None,
        s: None,
    };
    let plan = RiccatiPlan::build(
        problem,
        RiccatiKind::Discrete,
        RiccatiOptions::new(),
        ExecutionContext::sequential_native(),
    )?;
    // A scalar equation with positive costs uses the closed form, without scaling.
    assert_eq!(plan.state_scaling(), &[1.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples assert successful results for fixed valid inputs"
)]
fn equations_riccati_riccati_plan() -> Result<(), SciOxideError> {
    use scioxide_linalg::{Matrix, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem};
    let one = Matrix::<f64>::identity(1)?;
    let zero = Matrix::<f64>::zeros(1, 1)?;
    let problem = RiccatiProblem {
        a: zero.view(),
        b: one.view(),
        q: one.view(),
        r: one.view(),
        e: None,
        s: None,
    };
    let plan = RiccatiPlan::new(problem, RiccatiKind::Continuous, RiccatiOptions::new())?;
    let result = plan.solve()?;
    // With A = 0 and B = Q = R = 1 the CARE is 1 − x² = 0; the stabilizing root is 1.
    assert!((result.solution().unwrap().view().get(0, 0)? - 1.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn functions_condition_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
        MatrixFunctionWorkspaceRequest,
    };
    let a = Matrix::from_vec(1, 1, vec![2.0_f64])?;
    let plan = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &MatrixFunctionOptions::new())?;
    let mut workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Condition)?;
    let mut value = Matrix::<f64>::zeros(1, 1)?;
    let report = plan.condition_into(
        value.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // For scalars the absolute condition number of exp at a is exp(a).
    assert!((report.absolute() - 2.0_f64.exp()).abs() < 1e-12);
    assert!((value.view().get(0, 0)? - 2.0_f64.exp()).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn functions_frechet_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
        MatrixFunctionWorkspaceRequest,
    };
    let a = Matrix::from_vec(1, 1, vec![2.0_f64])?;
    let plan = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &MatrixFunctionOptions::new())?;
    let mut workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Frechet)?;
    let e = Matrix::from_vec(1, 1, vec![1.0_f64])?;
    let mut derivative = Matrix::<f64>::zeros(1, 1)?;
    plan.frechet_into(
        &e,
        derivative.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // For scalars L_exp(a, e) = exp(a) e.
    assert!((derivative.view().get(0, 0)? - 2.0_f64.exp()).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn functions_evaluate_into() -> Result<(), SciOxideError> {
    use scioxide_core::{Complex, ExecutionContext};
    use scioxide_linalg::{
        Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
        MatrixFunctionWorkspaceRequest,
    };
    let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    let plan = MatrixFunctionPlan::new(&a, MatrixFunction::Sqrt, &MatrixFunctionOptions::new())?;
    let mut workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Evaluate)?;
    let mut root = Matrix::<Complex<f64>>::zeros(1, 1)?;
    plan.evaluate_into(
        root.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // The principal square root of −1 is i.
    assert!((root.view().get(0, 0)? - Complex::new(0.0, 1.0)).norm() < 1e-15);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn functions_build() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan};
    let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    let options = MatrixFunctionOptions::new();
    // −1 lies on the branch cut of the logarithm, but √−1 = i is defined.
    let sqrt = MatrixFunctionPlan::build(
        &a,
        MatrixFunction::Sqrt,
        &options,
        ExecutionContext::sequential_native(),
    );
    assert!(sqrt.is_ok());
    let log = MatrixFunctionPlan::build(
        &a,
        MatrixFunction::Log,
        &options,
        ExecutionContext::sequential_native(),
    );
    assert!(log.is_err());
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn functions_reserved_plan() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
        MatrixFunctionWorkspaceRequest,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.5, 0.0, 2.0])?;
    let plan = MatrixFunctionPlan::new(&a, MatrixFunction::Log, &MatrixFunctionOptions::new())?;
    let mut workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Evaluate)?;
    let mut log = Matrix::<f64>::zeros(2, 2)?;
    let report = plan.evaluate_into(
        log.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // log([[λ, b], [0, λ]]) = [[log λ, b/λ], [0, log λ]].
    assert!((log.view().get(0, 0)? - 2.0_f64.ln()).abs() < 1e-14);
    assert!((log.view().get(0, 1)? - 0.25).abs() < 1e-14);
    assert!(report.work() > 0);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn iterative_arnoldi_krylov_recycle_space() -> Result<(), SciOxideError> {
    use scioxide_core::{Budget, ExecutionContext, ProblemToken, ResidualTolerance};
    use scioxide_linalg::{
        GcrotSolver, IterativeWorkspace, IterativeWorkspaceRequest, KrylovRecycleSpace, Matrix,
        Preconditioning, Vector,
    };
    let a = Matrix::from_vec(2, 2, vec![2.0, 1.0, 0.0, 3.0])?;
    let a = a.view().try_finite()?;
    let epoch = ProblemToken::new(1);
    let mut recycle = KrylovRecycleSpace::new(2, 2, epoch)?;
    let solver = GcrotSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(10)?);
    let mut workspace = IterativeWorkspace::new();
    workspace.try_reserve(&solver.workspace_requirements(
        &IterativeWorkspaceRequest::operator(&a)?.with_recycle_space(&recycle)?,
    )?)?;
    let mut x = Vector::zeros(2)?;
    let report = solver.solve_recycled_into(
        &a,
        &[3.0, 3.0][..],
        Preconditioning::None,
        epoch,
        &mut recycle,
        x.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert!(report.is_success());
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn iterative_bicgstab_bicgstab_solver() -> Result<(), SciOxideError> {
    use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
    use scioxide_linalg::{
        BicgstabSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination,
        Matrix, Preconditioning, Vector,
    };
    // A = I, so the solution is x = b.
    let a = Matrix::<f64>::identity(2)?;
    let operator = a.view().try_finite()?;
    let solver = BicgstabSolver::new(
        ResidualTolerance::new(1e-10, 0.0)?,
        Budget::try_from(20_usize)?,
    );
    let request = IterativeWorkspaceRequest::operator(&operator)?;
    let requirements = solver.workspace_requirements(&request)?;
    let mut workspace = IterativeWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let rhs = Vector::from_vec(vec![1.0, 2.0]);
    let mut solution = Vector::zeros(2)?;
    let report = solver.solve_into(
        &operator,
        &rhs,
        Preconditioning::None,
        solution.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.termination(), LinearSolveTermination::Converged);
    assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn iterative_cg_cg_solver() -> Result<(), SciOxideError> {
    use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
    use scioxide_linalg::{
        CgSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
        Preconditioning, Vector,
    };
    // A = I, so the solution is x = b.
    let a = Matrix::<f64>::identity(2)?;
    let operator = a.view().try_finite()?;
    let solver = CgSolver::new(
        ResidualTolerance::new(1e-10, 0.0)?,
        Budget::try_from(20_usize)?,
    );
    let request = IterativeWorkspaceRequest::operator(&operator)?;
    let requirements = solver.workspace_requirements(&request)?;
    let mut workspace = IterativeWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let rhs = Vector::from_vec(vec![1.0, 2.0]);
    let mut solution = Vector::zeros(2)?;
    let report = solver.solve_into(
        &operator,
        &rhs,
        Preconditioning::None,
        solution.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.termination(), LinearSolveTermination::Converged);
    assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn iterative_gmres_gmres_solver() -> Result<(), SciOxideError> {
    use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
    use scioxide_linalg::{
        GmresSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
        Preconditioning, Vector,
    };
    // A = I, so the solution is x = b.
    let a = Matrix::<f64>::identity(2)?;
    let operator = a.view().try_finite()?;
    let solver = GmresSolver::new(
        ResidualTolerance::new(1e-10, 0.0)?,
        Budget::try_from(20_usize)?,
    );
    let request = IterativeWorkspaceRequest::operator(&operator)?;
    let requirements = solver.workspace_requirements(&request)?;
    let mut workspace = IterativeWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let rhs = Vector::from_vec(vec![1.0, 2.0]);
    let mut solution = Vector::zeros(2)?;
    let report = solver.solve_into(
        &operator,
        &rhs,
        Preconditioning::None,
        solution.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.termination(), LinearSolveTermination::Converged);
    assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn iterative_stationary_richardson_solver() -> Result<(), SciOxideError> {
    use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
    use scioxide_linalg::{
        IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
        Preconditioning, RichardsonSolver, Vector,
    };
    // A = I, so the solution is x = b.
    let a = Matrix::<f64>::identity(2)?;
    let operator = a.view().try_finite()?;
    let solver = RichardsonSolver::new(
        ResidualTolerance::new(1e-10, 0.0)?,
        Budget::try_from(20_usize)?,
        scioxide_linalg::RelaxationWeight::new(1.0)?,
    );
    let request = IterativeWorkspaceRequest::operator(&operator)?;
    let requirements = solver.workspace_requirements(&request)?;
    let mut workspace = IterativeWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let rhs = Vector::from_vec(vec![1.0, 2.0]);
    let mut solution = Vector::zeros(2)?;
    let report = solver.solve_into(
        &operator,
        &rhs,
        Preconditioning::None,
        solution.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.termination(), LinearSolveTermination::Converged);
    assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_hermitian_rank_two_k_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::{Complex, Triangle};
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixView, MatrixViewMut,
        StructuredOperation,
    };
    // C := A Bᴴ + B Aᴴ on the lower triangle for A = [1, i]ᵀ and B = [1, 1]ᵀ.
    let (zero, one) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 0.0));
    let (a, b) = ([one, Complex::new(0.0, 1.0)], [one, one]);
    let mut c = [zero; 4];
    MatrixViewMut::hermitian_rank_two_k_in_place(
        MatrixView::from_slice(2, 1, &a)?,
        MatrixView::from_slice(2, 1, &b)?,
        Triangle::Lower,
        StructuredOperation::Normal,
        one,
        zero,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(
        [c[0], c[2], c[3]],
        [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0), zero]
    );
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_hermitian_rank_k_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::{Complex, Triangle};
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixView, MatrixViewMut,
        StructuredOperation,
    };
    // C := A Aᴴ on the lower triangle for A = [1, i]ᵀ.
    let (zero, one) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 0.0));
    let a = [one, Complex::new(0.0, 1.0)];
    let mut c = [zero; 4];
    MatrixViewMut::hermitian_rank_k_in_place(
        MatrixView::from_slice(2, 1, &a)?,
        Triangle::Lower,
        StructuredOperation::Normal,
        one,
        zero,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!([c[0], c[2], c[3]], [one, Complex::new(0.0, 1.0), one]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_symmetric_rank_two_k_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::Triangle;
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixView, MatrixViewMut,
        StructuredOperation,
    };
    // C := A Bᵀ + B Aᵀ on the lower triangle.
    let (a, b) = ([1.0_f64, 2.0], [1.0, 1.0]);
    let mut c = [0.0, 7.0, 0.0, 0.0];
    MatrixViewMut::symmetric_rank_two_k_in_place(
        MatrixView::from_slice(2, 1, &a)?,
        MatrixView::from_slice(2, 1, &b)?,
        Triangle::Lower,
        StructuredOperation::Normal,
        1.0,
        0.0,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(c, [2.0, 7.0, 3.0, 4.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_symmetric_rank_k_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::Triangle;
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixView, MatrixViewMut,
        StructuredOperation,
    };
    // C := A Aᵀ on the lower triangle; the upper 7.0 is unchanged.
    let a = [1.0_f64, 2.0];
    let mut c = [0.0, 7.0, 0.0, 0.0];
    MatrixViewMut::symmetric_rank_k_in_place(
        MatrixView::from_slice(2, 1, &a)?,
        Triangle::Lower,
        StructuredOperation::Normal,
        1.0,
        0.0,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(c, [1.0, 7.0, 2.0, 4.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_triangular_solve_mat_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixSide, MatrixView,
        MatrixViewMut, StructuredOperation,
    };
    let t = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    let mut b = [2.0, 4.0];
    t.triangular_solve_mat_in_place(
        MatrixSide::Left,
        Triangle::Lower,
        DiagonalKind::Explicit,
        StructuredOperation::Normal,
        1.0,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 1, &mut b)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(b, [1.0, 1.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_triangular_mul_mat_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixSide, MatrixView,
        MatrixViewMut, StructuredOperation,
    };
    let t = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    let mut b = [1.0, 1.0];
    t.triangular_mul_mat_in_place(
        MatrixSide::Left,
        Triangle::Lower,
        DiagonalKind::Explicit,
        StructuredOperation::Normal,
        2.0,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 1, &mut b)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(b, [4.0, 8.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_hermitian_mul_mat_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::{Complex, Triangle};
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixSide, MatrixView,
        MatrixViewMut,
    };
    let (zero, one) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 0.0));
    let a = [
        Complex::new(2.0, 0.0),
        zero,
        Complex::new(1.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let identity = [one, zero, zero, one];
    let mut c = [zero; 4];
    MatrixView::from_slice(2, 2, &a)?.hermitian_mul_mat_in_place(
        MatrixView::from_slice(2, 2, &identity)?,
        MatrixSide::Left,
        Triangle::Lower,
        one,
        zero,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(c[1], Complex::new(1.0, -1.0));
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn kernel_matrix_symmetric_mul_mat_in_place() -> Result<(), SciOxideError> {
    use scioxide_core::Triangle;
    use scioxide_linalg::{
        LinalgExecutionContext, LinalgExecutionOptions, LinalgWorkspace, MatrixSide, MatrixView,
        MatrixViewMut,
    };
    let a = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    let b = MatrixView::from_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])?;
    let mut c = [1.0; 4];
    a.symmetric_mul_mat_in_place(
        b,
        MatrixSide::Left,
        Triangle::Lower,
        2.0,
        1.0,
        &LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c)?,
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )?;
    assert_eq!(c, [5.0, 3.0, 3.0, 7.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn least_squares_trust_region_solve_trust_region_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::*;
    let a = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0])?;
    let options = LinalgExecutionOptions::new();
    let factor = ThinSvdDecomposition::compute(&a)?;
    let mut out = [0.0; 2];
    let report = factor.solve_trust_region_into(
        &[2.0][..],
        0.5,
        0.0,
        RankPolicy::Exact,
        &options,
        VectorViewMut::from_slice(&mut out),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )?;
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    // The step lies along the right singular vector (1, 1) / √2.
    assert!((out[0] - out[1]).abs() < 1e-14);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn nullspace_prepare_rhs_into() -> Result<(), SciOxideError> {
    use scioxide_core::{BlockVectorView, ExecutionContext};
    use scioxide_linalg::{
        BorderedSystem, BorderedSystemOptions, BorderedSystemWorkspaceRequest, CompatibilityPolicy,
        Matrix, VectorViewMut,
    };
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, -1.0, -1.0, 1.0])?;
    let operator = a.view().try_finite()?;
    let mode = [0.5_f64.sqrt(); 2];
    let basis = BlockVectorView::new(&mode, 2, 1)?;
    let options = BorderedSystemOptions::new();
    let system = BorderedSystem::new(&operator, basis, basis, 2.0, &options)?;
    let mut workspace = system.workspace(BorderedSystemWorkspaceRequest::PrepareRhs)?;
    let mut load = [0.0; 3];
    system.prepare_rhs_into(
        &[1.0, -1.0][..],
        &[0.0][..],
        CompatibilityPolicy::RequireCompatible,
        VectorViewMut::from_slice(&mut load),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let diagnose = system.workspace_requirements(BorderedSystemWorkspaceRequest::Diagnose)?;
    workspace.try_reserve(&diagnose)?;
    let report = system.diagnose_scoped(
        &[1.0, -1.0][..],
        load.as_slice(),
        &[0.5, -0.5][..],
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // (1, 1)ᵀ (0.5, −0.5) = 0 matches the gauge coordinate c = 0.
    assert!(report.gauge_defect() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn nullspace_bordered_system() -> Result<(), SciOxideError> {
    use scioxide_core::BlockVectorView;
    use scioxide_linalg::{BorderedSystem, BorderedSystemOptions, CompatibilityPolicy, Matrix};
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, -1.0, -1.0, 1.0])?;
    let operator = a.view().try_finite()?;
    let mode = [0.5_f64.sqrt(); 2];
    let basis = BlockVectorView::new(&mode, 2, 1)?;
    let options = BorderedSystemOptions::new();
    let system = BorderedSystem::new(&operator, basis, basis, 2.0, &options)?;
    let policy = CompatibilityPolicy::RequireCompatible;
    let load = system.prepare_rhs(&[1.0, -1.0][..], &[0.0][..], policy)?;
    // x = (0.5, −0.5) solves A x = (1, −1) and is orthogonal to the null vector (1, 1).
    let report = system.diagnose(&[1.0, -1.0][..], &load, &[0.5, -0.5][..])?;
    assert!(report.original_residual_norm() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn refinement_iterative_refinement() -> Result<(), SciOxideError> {
    use scioxide_core::ResidualTolerance;
    use scioxide_linalg::{
        IterativeRefinement, LuDecomposition, Matrix, RefinementOptions, RefinementProblem,
        RefinementSystem, Vector,
    };
    let a = Matrix::from_vec(1, 1, vec![2.0_f64])?;
    let low = Matrix::from_vec(1, 1, vec![2.0_f32])?;
    let factor = LuDecomposition::factor(&low)?;
    let inverse = factor.inverse_operator();
    let problem = RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None)?;
    let refinement =
        IterativeRefinement::new(RefinementOptions::new(ResidualTolerance::new(0.0, 1e-14)?))?;
    let (x, report) = refinement.refine(&problem, &[4.0][..], Vector::zeros(1)?.view())?;
    assert!(report.is_success());
    // 2 x = 4, so x = 2.
    assert!((x.as_slice()[0] - 2.0).abs() < 1e-14);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn sparse_functions_evaluate_complex() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{MatrixFunction, SparseFunctionOptions, SparseFunctionPlan};
    use scioxide_sparse::CsrMatrix;
    let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![-1.0_f64])?;
    let plan = SparseFunctionPlan::new(
        a.view(),
        MatrixFunction::Sqrt,
        &SparseFunctionOptions::new(),
    )?;
    let mut workspace = plan.workspace(())?;
    let mut root = plan.create_output::<scioxide_core::Complex<f64>>()?;
    plan.evaluate_into(
        a.view(),
        &mut root,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // The principal square root of −1 is i.
    assert!((root.values()[0].im - 1.0).abs() < 1e-15);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn sparse_functions_evaluate_real() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{SparsePowerOptions, SparsePowerPlan};
    use scioxide_sparse::CsrMatrix;
    let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 4.0])?;
    let plan = SparsePowerPlan::new(a.view(), -1, &SparsePowerOptions::new())?;
    let mut workspace = plan.workspace(())?;
    let mut inverse = plan.create_output()?;
    plan.evaluate_into(
        a.view(),
        &mut inverse,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // diag(2, 4)⁻¹ = diag(0.5, 0.25).
    assert_eq!(inverse.values(), &[0.5, 0.25]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_band_cholesky_factor() -> Result<(), SciOxideError> {
    use scioxide_core::Triangle;
    use scioxide_linalg::{
        BandedCholeskyDecomposition, HermitianBandMatrix, LinalgExecutionOptions, LinalgWorkspace,
    };
    // A = [4 2; 2 5] = L Lᵀ with L = [2 0; 1 2].
    let a = HermitianBandMatrix::new(2, 1, Triangle::Lower, vec![4.0, 2.0, 5.0, 0.0])?;
    let mut factor = BandedCholeskyDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    BandedCholeskyDecomposition::factor_into(
        &a,
        &LinalgExecutionOptions::new(),
        &mut factor,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(factor.lower()?.get(1, 0)?, 1.0);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_band_lu_factor() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        BandedLuDecomposition, BandedMatrix, LinalgExecutionOptions, LinalgWorkspace,
    };
    let a = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0, 1.0, 1.0, 1.0, 2.0, 0.0])?;
    let mut factor = BandedLuDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    let options = LinalgExecutionOptions::new();
    BandedLuDecomposition::factor_into(
        &a,
        &options,
        &mut factor,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(factor.order(), 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_direct_structured_workspace_request() -> Result<(), SciOxideError> {
    use scioxide_core::Capacity;
    use scioxide_linalg::{
        BandedLuDecomposition, LinalgExecutionOptions, LinalgWorkspace, StructuredWorkspaceRequest,
    };
    let options = LinalgExecutionOptions::new();
    let factor = BandedLuDecomposition::<f64>::workspace_requirements(
        &StructuredWorkspaceRequest::factor(8, 1, 2, &options),
    )?;
    let solve = BandedLuDecomposition::<f64>::workspace_requirements(
        &StructuredWorkspaceRequest::solve_matrix(8, 3, &options),
    )?;
    let mut workspace = LinalgWorkspace::<f64>::new();
    assert_eq!(workspace.check(&factor.max(&solve))?, Capacity::NeedsGrowth);
    workspace.try_reserve(&factor.max(&solve))?;
    assert_eq!(workspace.check(&solve)?, Capacity::Sufficient);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_displacement_solve_matrix_into() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        Matrix, StructuredMatrixView, StructuredOperation, StructuredSolveOptions,
        StructuredSolvePlan,
    };
    // A = 2 I as a circulant, so X = B / 2.
    let column = [2.0_f64, 0.0, 0.0];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::circulant(&column)?,
        &StructuredSolveOptions::new(),
    )?;
    let rhs = Matrix::from_vec(3, 2, vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0])?;
    let mut x = Matrix::zeros(3, 2)?;
    plan.solve_matrix_into(
        StructuredOperation::Transpose,
        &rhs,
        x.view_mut(),
        &mut scioxide_linalg::LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert!((x.view().get(2, 1)? - 2.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_displacement_structured_solve_plan() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        LinalgExecutionOptions, StructuredMatrixView, StructuredOperation, StructuredSolveOptions,
        StructuredSolvePlan, StructuredWorkspaceRequest, VectorViewMut,
    };
    // Nonsymmetric Toeplitz A = [4 2 0.5; 1 4 2; −1 1 4] with A [1, 1, 1]ᵀ = [6.5, 7, 4]ᵀ.
    let column = [4.0_f64, 1.0, -1.0];
    let row = [4.0, 2.0, 0.5];
    let matrix = StructuredMatrixView::toeplitz(&column, &row)?;
    let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new())?;
    let mut workspace = plan.workspace(&StructuredWorkspaceRequest::solve(
        3,
        &LinalgExecutionOptions::new(),
    ))?;
    let mut x = [0.0; 3];
    let report = plan.solve_into(
        StructuredOperation::Normal,
        &[6.5, 7.0, 4.0][..],
        VectorViewMut::from_slice(&mut x),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.gko_solves(), 1);
    assert!(report.backward_error() < 1e-12 && (x[0] - 1.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_displacement_structured_solve_report() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        LinalgExecutionOptions, StructuredMatrixView, StructuredOperation, StructuredSolveOptions,
        StructuredSolvePlan, StructuredWorkspaceRequest, VectorViewMut,
    };
    // Symmetric Toeplitz A = [2 0.5; 0.5 2] is solved by Levinson–Durbin recursion.
    let column = [2.0_f64, 0.5];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&column, &column)?,
        &StructuredSolveOptions::new(),
    )?;
    let request = StructuredWorkspaceRequest::solve(2, &LinalgExecutionOptions::new());
    let mut x = [0.0; 2];
    let report = plan.solve_into(
        StructuredOperation::Normal,
        &[2.5, 2.5][..],
        VectorViewMut::from_slice(&mut x),
        &mut plan.workspace(&request)?,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(report.levinson_solves(), 1);
    assert!(report.backward_error() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_block_diagonal_solve() -> Result<(), SciOxideError> {
    use scioxide_core::{AdjointOperator, Tolerance};
    use scioxide_linalg::{
        BlockDiagonalOperator, CompositeSolveOptions, LinalgWorkspace, LuDecomposition, Matrix,
        VectorView, VectorViewMut,
    };
    // diag(2, 2) x = [2, 4] has the solution x = [1, 2].
    let a = Matrix::from_vec(1, 1, vec![2.0_f64])?;
    let af = a.view().try_finite()?;
    let blocks: [&(dyn AdjointOperator<f64> + Sync); 2] = [&af, &af];
    let operator = BlockDiagonalOperator::new(&blocks)?;
    let inverse = LuDecomposition::factor(&a)?;
    let inverse = inverse.inverse_operator();
    let mut x = [0.0; 2];
    let options = CompositeSolveOptions::new().with_tolerance(Tolerance::new(0.0, 1e-12)?);
    let report = operator.solve_into(
        &[&inverse, &inverse],
        VectorView::from_slice(&[2.0, 4.0]),
        &options,
        VectorViewMut::from_slice(&mut x),
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert!(report.termination().is_success());
    assert_eq!(x, [1.0, 2.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_kronecker_solve() -> Result<(), SciOxideError> {
    use scioxide_core::Tolerance;
    use scioxide_linalg::{
        CompositeSolveOptions, KroneckerOperator, LinalgWorkspace, LuDecomposition, Matrix,
        VectorView, VectorViewMut,
    };
    // A = diag(2, 4), so A ⊗ A = diag(4, 8, 8, 16) and x = [1, 1, 1, 1].
    let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 4.0])?;
    let (fa, af) = (LuDecomposition::factor(&a)?, a.view().try_finite()?);
    let product = KroneckerOperator::new(&af, &af)?;
    let (inverse, rhs) = (fa.inverse_operator(), [4.0, 8.0, 8.0, 16.0]);
    let mut x = [0.0; 4];
    let report = product.solve_into(
        [&inverse, &inverse],
        VectorView::from_slice(&rhs),
        &CompositeSolveOptions::new().with_tolerance(Tolerance::new(0.0, 1e-12)?),
        VectorViewMut::from_slice(&mut x),
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert!(report.termination().is_success());
    assert_eq!(x, [1.0; 4]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_tridiagonal_solve_into() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        CyclicTridiagonalPlan, LinalgExecutionOptions, StructuredOperation,
        StructuredWorkspaceRequest, TridiagonalMatrixView, VectorViewMut,
    };
    let options = LinalgExecutionOptions::new();
    let plan = CyclicTridiagonalPlan::new(
        TridiagonalMatrixView::new(&[-1.0; 2], &[4.0; 3], &[-1.0; 2])?,
        -1.0,
        -1.0,
        &options,
    )?;
    let mut x = [0.0; 3];
    let mut workspace = plan.workspace(&StructuredWorkspaceRequest::solve(3, &options))?;
    let acceptance = scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact);
    // A is symmetric with row sums 2, so Aᴴ x = [2, 2, 2] has x = [1, 1, 1].
    let report = plan.solve_into(
        StructuredOperation::Adjoint,
        &[2.0, 2.0, 2.0][..],
        &acceptance,
        VectorViewMut::from_slice(&mut x),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert!(report.termination().is_success());
    assert!(x.iter().all(|value| (value - 1.0_f64).abs() < 1e-12));
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_tridiagonal_workspace_requirements() -> Result<(), SciOxideError> {
    use scioxide_core::Capacity;
    use scioxide_linalg::{
        LinalgExecutionOptions, LinalgWorkspace, StructuredWorkspaceRequest,
        TridiagonalLuDecomposition,
    };
    let options = LinalgExecutionOptions::new();
    let factor = TridiagonalLuDecomposition::<f64>::workspace_requirements(
        &StructuredWorkspaceRequest::factor(64, 1, 1, &options),
    )?;
    let solve = TridiagonalLuDecomposition::<f64>::workspace_requirements(
        &StructuredWorkspaceRequest::solve(64, &options),
    )?;
    let mut workspace = LinalgWorkspace::<f64>::new();
    workspace.try_reserve(&factor.max(&solve))?;
    assert_eq!(workspace.check(&factor)?, Capacity::Sufficient);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn structured_utilities_apply_operator_matrix_into() -> Result<(), SciOxideError> {
    use scioxide_linalg::{LinalgWorkspace, Matrix, MatrixView, apply_operator_matrix_into};
    // [1 2; 3 4] [1, 1]ᵀ = [3, 7]ᵀ.
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
    let finite = a.view().try_finite()?;
    let input = MatrixView::from_slice(2, 1, &[1.0, 1.0])?;
    let mut output = Matrix::zeros(2, 1)?;
    apply_operator_matrix_into(
        &finite,
        input,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        output.view_mut(),
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    assert_eq!(output.as_slice(), &[3.0, 7.0]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn subspace_orthogonal_procrustes_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        LinalgExecutionOptions, LinalgWorkspace, Matrix, ProcrustesWorkspaceRequest, RankPolicy,
        orthogonal_procrustes_into,
    };
    let a = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let b = Matrix::from_vec(3, 2, vec![0.0_f64, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let options = LinalgExecutionOptions::new();
    let request = ProcrustesWorkspaceRequest::new(3, 2, &options);
    let mut workspace = LinalgWorkspace::<f64>::workspace(&request)?;
    let mut rotation = Matrix::empty();
    let report = orthogonal_procrustes_into(
        &a,
        &b,
        RankPolicy::Exact,
        &options,
        &mut rotation,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // B is A with its columns swapped, so Q is the 2 × 2 swap [[0, 1], [1, 0]].
    assert!(report.residual_norm() < 1e-12);
    assert!((rotation.view().get(0, 1)? - 1.0).abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn subspace_subspace_angles_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy,
        SubspaceAnglesWorkspaceRequest, subspace_angles_into,
    };
    let a = Matrix::from_vec(3, 1, vec![1.0_f64, 0.0, 0.0])?;
    let b = Matrix::from_vec(3, 2, vec![0.0_f64, 1.0, 1.0, 0.0, 0.0, 0.0])?;
    let options = LinalgExecutionOptions::new();
    let request = SubspaceAnglesWorkspaceRequest::new(3, 1, 2, &options);
    let mut workspace = LinalgWorkspace::<f64>::workspace(&request)?;
    let mut angles = Vec::new();
    subspace_angles_into(
        &a,
        &b,
        RankPolicy::Exact,
        &options,
        &mut angles,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    // span(a) = span(e₀) lies inside span(b) = span(e₀, e₁).
    assert!(angles[0].abs() < 1e-12);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn subspace_null_space_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy, null_space_into,
    };
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1.0, 1.0])?;
    let mut basis = Matrix::empty();
    let report = null_space_into(
        &a,
        RankPolicy::DimensionScaledEpsilon,
        &LinalgExecutionOptions::new(),
        &mut basis,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!((report.rank().rank(), basis.cols()), (1, 1));
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn subspace_orth_into() -> Result<(), SciOxideError> {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::{
        LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy, SubspaceBasisWorkspaceRequest,
        orth_into,
    };
    let a = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0])?;
    let options = LinalgExecutionOptions::new();
    let mut workspace =
        LinalgWorkspace::<f64>::workspace(&SubspaceBasisWorkspaceRequest::orth(3, 2, &options))?;
    let mut basis = Matrix::empty();
    let report = orth_into(
        &a,
        RankPolicy::Exact,
        &options,
        &mut basis,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(
        (report.rank().rank(), basis.rows(), basis.cols()),
        (2, 3, 2)
    );
    Ok::<(), scioxide_core::SciOxideError>(())
}

pub fn main() -> Result<(), SciOxideError> {
    actions_cancelled_norm_estimate()?;
    #[cfg(feature = "blas")]
    blas_provider_dispatch()?;
    decompose_householder_trust_region_solve_trust_region_into()?;
    decompose_lu_inverse_into()?;
    decompose_minimum_norm_householder_minimum_norm_plan()?;
    eigen_general_decompose_into()?;
    eigen_general_compute_into()?;
    eigen_generalized_decompose_into()?;
    eigen_generalized_compute_into()?;
    eigen_hermitian_tridiagonal_values()?;
    eigen_hermitian_tridiagonal_vectors()?;
    eigen_hermitian_decompose_into()?;
    eigen_hermitian_compute_into()?;
    eigen_qz_decompose_into()?;
    eigen_qz_reorder_reorder_in_place()?;
    eigen_qz_types_generalized_schur_workspace_request()?;
    eigen_schur_decompose_into()?;
    eigen_schur_reorder_reorder_in_place()?;
    equations_riccati_solve_into()?;
    equations_riccati_build()?;
    equations_riccati_riccati_plan()?;
    functions_condition_into()?;
    functions_frechet_into()?;
    functions_evaluate_into()?;
    functions_build()?;
    functions_reserved_plan()?;
    iterative_arnoldi_krylov_recycle_space()?;
    iterative_bicgstab_bicgstab_solver()?;
    iterative_cg_cg_solver()?;
    iterative_gmres_gmres_solver()?;
    iterative_stationary_richardson_solver()?;
    kernel_matrix_hermitian_rank_two_k_in_place()?;
    kernel_matrix_hermitian_rank_k_in_place()?;
    kernel_matrix_symmetric_rank_two_k_in_place()?;
    kernel_matrix_symmetric_rank_k_in_place()?;
    kernel_matrix_triangular_solve_mat_in_place()?;
    kernel_matrix_triangular_mul_mat_in_place()?;
    kernel_matrix_hermitian_mul_mat_in_place()?;
    kernel_matrix_symmetric_mul_mat_in_place()?;
    least_squares_trust_region_solve_trust_region_into()?;
    nullspace_prepare_rhs_into()?;
    nullspace_bordered_system()?;
    refinement_iterative_refinement()?;
    sparse_functions_evaluate_complex()?;
    sparse_functions_evaluate_real()?;
    structured_band_cholesky_factor()?;
    structured_band_lu_factor()?;
    structured_direct_structured_workspace_request()?;
    structured_displacement_solve_matrix_into()?;
    structured_displacement_structured_solve_plan()?;
    structured_displacement_structured_solve_report()?;
    structured_block_diagonal_solve()?;
    structured_kronecker_solve()?;
    structured_tridiagonal_solve_into()?;
    structured_tridiagonal_workspace_requirements()?;
    structured_utilities_apply_operator_matrix_into()?;
    subspace_orthogonal_procrustes_into()?;
    subspace_subspace_angles_into()?;
    subspace_null_space_into()?;
    subspace_orth_into()?;
    Ok(())
}
