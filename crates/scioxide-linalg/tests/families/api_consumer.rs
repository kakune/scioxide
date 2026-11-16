//! External-consumer signature and getter contracts.
use scioxide_core::{
    AttemptRecord, Capacity, ExecutionContext, LinearOperator, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::*;

// Lifecycle entry points of the iterative, action and refinement forms.
const _: fn() -> IterativeWorkspace<f64> = IterativeWorkspace::new;
const _: fn(
    &IterativeWorkspace<f64>,
    &IterativeRequirements<f64>,
) -> Result<Capacity, SciOxideError> = IterativeWorkspace::check;
const _: fn(
    &mut IterativeWorkspace<f64>,
    &IterativeRequirements<f64>,
) -> Result<(), SciOxideError> = IterativeWorkspace::try_reserve;
const _: for<'a> fn(
    &'a IterativeWorkspace<f64>,
) -> Option<&'a AttemptRecord<IterativeProgress<f64>>> = IterativeWorkspace::last_attempt;
const _: for<'a> fn(&'a IterativeWorkspace<f64>) -> Option<IterativeAccepted<'a, f64>> =
    IterativeWorkspace::last_accepted;
const _: for<'a> fn(
    &'a ExponentialActionWorkspace<f64>,
) -> Option<scioxide_linalg::ExponentialActionAccepted<'a, f64>> =
    ExponentialActionWorkspace::last_accepted;
const _: for<'a> fn(
    &'a FunctionActionWorkspace<f64>,
) -> Option<scioxide_linalg::FunctionActionAccepted<'a, f64>> =
    FunctionActionWorkspace::last_accepted;
const _: fn(&SchurDecomposition<f64>) -> Option<scioxide_linalg::EigenReport> =
    SchurDecomposition::report;
const _: fn(&GeneralEigenDecomposition<f64>) -> Option<scioxide_linalg::EigenReport> =
    GeneralEigenDecomposition::report;
const _: fn(&HermitianEigenDecomposition<f64>) -> Option<scioxide_linalg::EigenReport> =
    HermitianEigenDecomposition::report;
const _: fn() -> HermitianEigenFirstComponents<f64> = HermitianEigenFirstComponents::empty;
const _: for<'a> fn(&'a HermitianEigenFirstComponents<f64>) -> &'a [f64] =
    HermitianEigenFirstComponents::first_components;
const _: fn(&HermitianEigenFirstComponents<f64>) -> Option<EigenReport> =
    HermitianEigenFirstComponents::report;
const _: fn(&GeneralizedHermitianEigenDecomposition<f64>) -> Option<scioxide_linalg::EigenReport> =
    GeneralizedHermitianEigenDecomposition::report;
const _: fn(&GeneralizedSchurDecomposition<f64>) -> Option<GeneralizedSchurReport> =
    GeneralizedSchurDecomposition::report;
const _: fn(&dyn LinearOperator<f64>) -> Result<IterativeWorkspaceRequest, SciOxideError> =
    IterativeWorkspaceRequest::operator;
const _: fn(f64) -> Result<RelaxationWeight<f64>, SciOxideError> = RelaxationWeight::new;
const _: fn(f64) -> Result<SorWeight<f64>, SciOxideError> = SorWeight::new;
const _: fn() -> OneNormEstimator = OneNormEstimator::new;
const _: fn(f64) -> ExponentialAction<f64> = ExponentialAction::new;
const _: fn(ResidualTolerance<f64>) -> RefinementOptions<f64> = RefinementOptions::new;
const _: fn() -> RefinementWorkspace<f32, f64> = RefinementWorkspace::new;

/// Argument order of the borrowed forms: inputs, state, workspace, context.
#[allow(dead_code)]
fn borrowed_forms(
    a: &dyn LinearOperator<f64>,
    b: VectorView<'_, f64>,
    x: VectorViewMut<'_, f64>,
    workspace: &mut IterativeWorkspace<f64>,
) -> Result<IterativeSolveReport<f64>, SciOxideError> {
    let solver = GmresSolver::new(ResidualTolerance::new(0., 1e-10)?, 8_usize.try_into()?);
    solver.solve_into(
        a,
        b,
        Preconditioning::None,
        x,
        workspace,
        ExecutionContext::sequential(),
    )
}

// Direct factors: borrowing, empty construction and one explicit input interpretation.
const _: fn() -> LdlDecomposition<f64> = LdlDecomposition::empty;
const _: for<'a> fn(&'a LuDecomposition<f64>) -> &'a [usize] = LuDecomposition::permutation;
const _: for<'a> fn(&'a LdlDecomposition<f64>) -> &'a Matrix<f64> = LdlDecomposition::l_matrix;
const _: for<'a> fn(&'a CholeskyDecomposition<f64>) -> &'a Matrix<f64> =
    CholeskyDecomposition::l_matrix;
const _: fn(&LuDecomposition<f64>) -> Result<Matrix<f64>, SciOxideError> =
    LuDecomposition::to_l_matrix;
const _: fn(&LuDecomposition<f64>) -> Result<Matrix<f64>, SciOxideError> =
    LuDecomposition::to_u_matrix;
const _: fn(&HouseholderDecomposition<f64>) -> Result<Matrix<f64>, SciOxideError> =
    HouseholderDecomposition::to_triangular_matrix;
const _: for<'a> fn(&'a CholeskyDecomposition<f64>) -> Option<&'a HermitianProjectionReport<f64>> =
    CholeskyDecomposition::projection_report;
const _: for<'a> fn(&'a LdlDecomposition<f64>) -> Option<&'a HermitianProjectionReport<f64>> =
    LdlDecomposition::projection_report;
const _: HermitianInput<f64> = HermitianInput::Exact;

// Dense/BLAS results and infallible capability configuration.
#[allow(dead_code)]
fn borrowed_diagonal<'a>(
    matrix: MatrixView<'a, f64>,
) -> Result<VectorView<'a, f64>, SciOxideError> {
    matrix.try_diag()
}
#[cfg(feature = "blas")]
const _: fn(BlasCapabilities, usize) -> BlasCapabilities = BlasCapabilities::with_minimum_alignment;
#[cfg(feature = "blas")]
const _: fn(&BlasCapabilities) -> usize = BlasCapabilities::minimum_alignment;

#[allow(dead_code)]
fn dense_reduction_forms(
    input: VectorView<'_, f64>,
) -> Result<GivensRotationResult<f64>, SciOxideError> {
    let cx = ExecutionContext::sequential;
    let _ = input.dotu_scoped(input, &scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    let _ = input.dotc_left_scoped(input, &scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    let _ = input.dotc_left_scoped(input, &scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    let _ = input.norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    let _ = input.asum_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    let _ = input.iamax_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), cx())?;
    givens_rotation_scoped(
        3.0,
        4.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        cx(),
    )
}

// Unified direct-factor policy, determinant and inverse-operator surface.
const _: fn(&LuDecomposition<f64>) -> usize = LuDecomposition::order;
const _: fn(&CholeskyDecomposition<f64>) -> usize = CholeskyDecomposition::order;
const _: fn(&LdlDecomposition<f64>) -> usize = LdlDecomposition::order;
const _: fn(&TridiagonalLuDecomposition<f64>) -> usize = TridiagonalLuDecomposition::order;
const _: fn(SignLogDeterminant<f64>) -> f64 = SignLogDeterminant::log_abs;
const _: fn(&CholeskyDecomposition<f64>) -> Result<SignLogDeterminant<f64>, SciOxideError> =
    CholeskyDecomposition::slogdet;
const _: fn(&LdlDecomposition<f64>) -> Result<SignLogDeterminant<f64>, SciOxideError> =
    LdlDecomposition::slogdet;
const _: fn(
    &LdlDecomposition<f64>,
    RankPolicy<f64>,
) -> Result<scioxide_sparse::LdlInertia, SciOxideError> = LdlDecomposition::inertia;

#[allow(dead_code)]
fn direct_operator_forms(
    lu: &LuDecomposition<f64>,
    cholesky: &CholeskyDecomposition<f64>,
    ldl: &LdlDecomposition<f64>,
) {
    fn traits<
        T: scioxide_core::LinearOperator<f64>
            + scioxide_core::AdjointOperator<f64>
            + scioxide_core::Preconditioner<f64>,
    >(
        _: T,
    ) {
    }
    traits(lu.inverse_operator());
    traits(cholesky.inverse_operator());
    traits(ldl.inverse_operator());
}

#[allow(dead_code)]
fn direct_policy_forms(
    factor: &CholeskyDecomposition<f64>,
    rhs: VectorView<'_, f64>,
    output: VectorViewMut<'_, f64>,
    workspace: &mut LinalgWorkspace<'_, f64>,
) -> Result<DirectSolveReport<f64>, SciOxideError> {
    let options = DirectSolveOptions::new(RankPolicy::Exact)
        .with_residual(Estimation::Estimate)
        .with_execution_options(LinalgExecutionOptions::new());
    let _: DirectSolveResult<Vector<f64>, f64> = factor.solve_with_options(rhs, &options)?;
    factor.solve_into(
        rhs,
        &options,
        output,
        workspace,
        ExecutionContext::sequential(),
    )
}

// Structured plans share the factor request shape and take the operation per solve.
#[allow(dead_code)]
fn structured_plan_forms(
    compact: &StructuredSolvePlan<'_, f64>,
    cyclic: &CyclicTridiagonalPlan<f64>,
    rhs: MatrixView<'_, f64>,
    output: scioxide_linalg::MatrixViewMut<'_, f64>,
    workspace: &mut LinalgWorkspace<'_, f64>,
) -> Result<StructuredSolveReport<f64>, SciOxideError> {
    let request = StructuredWorkspaceRequest::solve_matrix(
        rhs.rows(),
        rhs.cols(),
        &LinalgExecutionOptions::new(),
    );
    let _ = cyclic.workspace_requirements(&request)?;
    let _: DirectSolveResult<Matrix<f64>, f64> = cyclic.solve_matrix_with_options(
        StructuredOperation::Adjoint,
        rhs,
        &DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon),
    )?;
    workspace.try_reserve(&compact.workspace_requirements(&request)?)?;
    compact.solve_matrix_into(
        StructuredOperation::Transpose,
        rhs,
        output,
        workspace,
        ExecutionContext::sequential(),
    )
}

// Compiled for every exported immutable plan family, including borrowed plans.
const _: fn() = || {
    fn traits<T: Send + Sync + Clone>() {}
    fn numeric<'a, S: scioxide_core::NumericScalar>(_: &'a ())
    where
        scioxide_core::Complex<S::Real>: scioxide_core::NumericScalar<Real = S::Real>,
    {
        macro_rules! plans { ($($name:ident),+ $(,)?) => { $(traits::<$name<S>>();)+ }; }
        plans!(
            CgSolver,
            GmresSolver,
            BlockGmresSolver,
            BicgstabSolver,
            MinresSolver,
            RichardsonSolver,
            CgsSolver,
            TfqmrSolver,
            LgmresSolver,
            GcrotSolver,
            BicgSolver,
            QmrSolver,
            LsqrSolver,
            LsmrSolver,
            JacobiSolver,
            GaussSeidelSolver,
            SorSolver,
            BlockLgmresSolver,
            BlockGcrotSolver,
            HouseholderMinimumNormPlan,
            SylvesterPlan,
            LyapunovPlan,
            RiccatiPlan,
            SparsePowerPlan,
            CyclicTridiagonalPlan
        );
        traits::<IterativeRefinement<S, S>>();
        traits::<RefinementProblem<'a, S, S>>();
        traits::<BorderedSystem<'a, S>>();
        traits::<TriangularPlan<'a, S>>();
        traits::<StructuredSolvePlan<'a, S>>();
        traits::<SparseFunctionPlan<'a, S>>();
    }
    fn real<'a, R: scioxide_core::RealScalar>(_: &'a ())
    where
        scioxide_core::Complex<R>: scioxide_core::NumericScalar<Real = R>,
    {
        macro_rules! plans { ($($name:ident),+ $(,)?) => { $(traits::<$name<R>>();)+ }; }
        plans!(
            ExponentialAction,
            FunctionAction,
            PartialSpectrumSolver,
            LobpcgSolver,
            PartialSvdSolver,
            RandomizedLowRank
        );
        traits::<MatrixFunctionPlan<'a, R>>();
    }
    numeric::<f32>(&());
    numeric::<f64>(&());
    numeric::<scioxide_core::Complex<f32>>(&());
    numeric::<scioxide_core::Complex<f64>>(&());
    real::<f32>(&());
    real::<f64>(&());
    traits::<OneNormEstimator>();
    traits::<IterativeRefinement<f32, f64>>();
    traits::<IterativeRefinement<scioxide_core::Complex<f32>, scioxide_core::Complex<f64>>>();
    traits::<RefinementProblem<'_, f32, f64>>();
    traits::<RefinementProblem<'_, scioxide_core::Complex<f32>, scioxide_core::Complex<f64>>>();
};

// Getter signatures are checked from outside the crate, including borrowed views.
#[test]
fn report_and_output_getters_are_public() {
    type DirectVectorResult = DirectSolveResult<Vector<f64>, f64>;
    use scioxide_core::backend::ScalarKind;
    use scioxide_core::{Complex, PartialResultTrust, ReliableEstimate, ResidualReport};
    use scioxide_linalg::*;
    use scioxide_sparse::CsrMatrix;
    let _: for<'a> fn(&'a SubspaceReport<f64>) -> RankReport<f64> = SubspaceReport::rank;
    let _: for<'a> fn(&'a SubspaceReport<f64>) -> Option<ConditionEstimate<f64>> =
        SubspaceReport::condition;
    let _: for<'a> fn(&'a SubspaceBasis<f64>) -> &Matrix<f64> = SubspaceBasis::basis;
    let _: for<'a> fn(&'a SubspaceBasis<f64>) -> SubspaceReport<f64> = SubspaceBasis::report;
    let _: for<'a> fn(&'a SubspaceAnglesReport<f64>) -> SubspaceReport<f64> =
        SubspaceAnglesReport::first;
    let _: for<'a> fn(&'a SubspaceAnglesReport<f64>) -> SubspaceReport<f64> =
        SubspaceAnglesReport::second;
    let _: for<'a> fn(&'a SubspaceAngles<f64>) -> &[f64] = SubspaceAngles::angles;
    let _: for<'a> fn(&'a SubspaceAngles<f64>) -> SubspaceAnglesReport<f64> =
        SubspaceAngles::report;
    let _: for<'a> fn(&'a ProcrustesReport<f64>) -> f64 = ProcrustesReport::scale;
    let _: for<'a> fn(&'a ProcrustesReport<f64>) -> f64 = ProcrustesReport::residual_norm;
    let _: for<'a> fn(&'a ProcrustesReport<f64>) -> SubspaceReport<f64> =
        ProcrustesReport::cross_product;
    let _: for<'a> fn(&'a ProcrustesReport<f64>) -> ProcrustesUniqueness =
        ProcrustesReport::uniqueness;
    let _: for<'a> fn(&'a ProcrustesAlignment<f64>) -> &Matrix<f64> =
        ProcrustesAlignment::transformation;
    let _: for<'a> fn(&'a ProcrustesAlignment<f64>) -> ProcrustesReport<f64> =
        ProcrustesAlignment::report;
    let _: for<'a> fn(&'a QrUpdateReport<f64>) -> QrRankDecision<f64> = QrUpdateReport::before;
    let _: for<'a> fn(&'a QrUpdateReport<f64>) -> QrRankDecision<f64> = QrUpdateReport::after;
    let _: for<'a> fn(&'a QrUpdateReport<f64>) -> bool = QrUpdateReport::rank_lost;
    let _: fn(HermitianProjectionReport<f64>) -> f64 =
        HermitianProjectionReport::absolute_frobenius;
    let _: fn(HermitianProjectionReport<f64>) -> f64 =
        HermitianProjectionReport::relative_frobenius;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> RankReport<f64> = GsvdReport::rank;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> Option<ConditionEstimate<f64>> =
        GsvdReport::condition;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> f64 = GsvdReport::retained_reciprocal_condition;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> f64 = GsvdReport::discarded_norm;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> f64 = GsvdReport::residual_a;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> f64 = GsvdReport::residual_b;
    let _: for<'a> fn(&'a GsvdReport<f64>) -> usize = GsvdReport::sweeps;
    let _: for<'a> fn(&'a CsdReport<f64>) -> f64 = CsdReport::input_unitarity_error;
    let _: for<'a> fn(&'a CsdReport<f64>) -> f64 = CsdReport::residual;
    let _: for<'a> fn(&'a CsdReport<f64>) -> f64 = CsdReport::orthogonality_error;
    let _: for<'a> fn(&'a CsdReport<f64>) -> usize = CsdReport::sweeps;
    let _: for<'a> fn(&'a PolarReport<f64>) -> RankReport<f64> = PolarReport::rank;
    let _: for<'a> fn(&'a PolarReport<f64>) -> Option<ConditionEstimate<f64>> =
        PolarReport::condition;
    let _: for<'a> fn(&'a PolarReport<f64>) -> bool = PolarReport::completion_nonunique;
    let _: for<'a> fn(&'a PolarReport<f64>) -> f64 = PolarReport::residual;
    let _: for<'a> fn(&'a PolarReport<f64>) -> f64 = PolarReport::orthogonality_error;
    let _: fn(CholeskyDefinitenessReport<f64>) -> RankReport<f64> =
        CholeskyDefinitenessReport::factor_rank;
    let _: fn(CholeskyDefinitenessReport<f64>) -> RankReport<f64> =
        CholeskyDefinitenessReport::effective_rank;
    let _: for<'a> fn(&'a GivensRotationResult<f64>) -> f64 = GivensRotationResult::r;
    let _: for<'a> fn(&'a GivensRotationResult<f64>) -> f64 = GivensRotationResult::z;
    let _: for<'a> fn(&'a GivensRotationResult<f64>) -> GivensRotation<f64> =
        GivensRotationResult::rotation;
    let _: for<'a> fn(&'a ModifiedRotationResult<f64>) -> f64 = ModifiedRotationResult::d1;
    let _: for<'a> fn(&'a ModifiedRotationResult<f64>) -> f64 = ModifiedRotationResult::d2;
    let _: for<'a> fn(&'a ModifiedRotationResult<f64>) -> f64 = ModifiedRotationResult::x1;
    let _: for<'a> fn(&'a ModifiedRotationResult<f64>) -> ModifiedRotation<f64> =
        ModifiedRotationResult::rotation;
    let _: for<'a> fn(&'a MatrixFunctionReport<f64>) -> usize = MatrixFunctionReport::work;
    let _: for<'a> fn(&'a MatrixFunctionReport<f64>) -> Option<f64> =
        MatrixFunctionReport::branch_distance;
    let _: for<'a> fn(&'a MatrixFunctionReport<f64>) -> f64 =
        MatrixFunctionReport::imaginary_defect;
    let _: for<'a> fn(&'a MatrixFunctionReport<f64>) -> scioxide_core::Scaled<f64> =
        MatrixFunctionReport::output_norm;
    let _: for<'a> fn(&'a MatrixFunctionConditionReport<f64>) -> f64 =
        MatrixFunctionConditionReport::absolute;
    let _: for<'a> fn(&'a MatrixFunctionConditionReport<f64>) -> Option<f64> =
        MatrixFunctionConditionReport::relative;
    let _: for<'a> fn(&'a MatrixFunctionConditionReport<f64>) -> usize =
        MatrixFunctionConditionReport::work;
    let _: for<'a> fn(&'a FunctionActionReport) -> SpectralTermination =
        FunctionActionReport::termination;
    let _: for<'a> fn(&'a FunctionActionReport) -> usize = FunctionActionReport::applications;
    let _: for<'a> fn(&'a FunctionActionReport) -> usize = FunctionActionReport::cycles;
    let _: for<'a> fn(&'a FunctionActionAccepted<'static, f64>) -> &'a [f64] =
        FunctionActionAccepted::estimated_errors;
    let _: for<'a> fn(&'a FunctionActionAccepted<'static, f64>) -> &'a [bool] =
        FunctionActionAccepted::completed;
    let _: for<'a> fn(&'a PartialSpectrumBuffer<f64>) -> &[Complex<f64>] =
        PartialSpectrumBuffer::eigenvalues;
    let _: for<'a> fn(
        &'a PartialSpectrumBuffer<f64>,
    ) -> scioxide_linalg::MatrixView<'a, Complex<f64>> = PartialSpectrumBuffer::eigenvectors;
    let _: for<'a> fn(&'a PartialSpectrumBuffer<f64>) -> &[f64] = PartialSpectrumBuffer::residuals;
    let _: for<'a> fn(&'a PartialSpectrumReport) -> SpectralTermination =
        PartialSpectrumReport::termination;
    let _: for<'a> fn(&'a PartialSpectrumReport) -> usize = PartialSpectrumReport::applications;
    let _: for<'a> fn(&'a PartialSpectrumReport) -> usize = PartialSpectrumReport::cycles;
    let _: for<'a> fn(&'a ExponentialActionReport) -> SpectralTermination =
        ExponentialActionReport::termination;
    let _: for<'a> fn(&'a ExponentialActionReport) -> usize = ExponentialActionReport::applications;
    let _: for<'a> fn(&'a ExponentialActionAccepted<'static, f64>) -> &'a [f64] =
        ExponentialActionAccepted::progress;
    let _: for<'a> fn(&'a ExponentialActionAccepted<'static, f64>) -> &'a [f64] =
        ExponentialActionAccepted::estimated_errors;
    let _: for<'a> fn(&'a ExponentialActionAccepted<'static, f64>) -> &'a [bool] =
        ExponentialActionAccepted::completed;
    let _: for<'a> fn(&'a LobpcgReport) -> SpectralTermination = LobpcgReport::termination;
    let _: for<'a> fn(&'a LobpcgReport) -> usize = LobpcgReport::applications;
    let _: for<'a> fn(&'a LobpcgReport) -> usize = LobpcgReport::iterations;
    let _: for<'a> fn(&'a LobpcgBuffer<f64>) -> &[f64] = LobpcgBuffer::eigenvalues;
    let _: for<'a> fn(&'a LobpcgBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        LobpcgBuffer::eigenvectors;
    let _: for<'a> fn(&'a LobpcgBuffer<f64>) -> &[f64] = LobpcgBuffer::residuals;
    let _: for<'a> fn(&'a PartialSvdBuffer<f64>) -> &[f64] = PartialSvdBuffer::singular_values;
    let _: for<'a> fn(&'a PartialSvdBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        PartialSvdBuffer::left_vectors;
    let _: for<'a> fn(&'a PartialSvdBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        PartialSvdBuffer::right_vectors;
    let _: for<'a> fn(&'a PartialSvdBuffer<f64>) -> &[[f64; 2]] = PartialSvdBuffer::residuals;
    let _: for<'a> fn(&'a PartialSvdReport) -> SpectralTermination = PartialSvdReport::termination;
    let _: for<'a> fn(&'a PartialSvdReport) -> usize = PartialSvdReport::applications;
    let _: for<'a> fn(&'a PartialSvdReport) -> usize = PartialSvdReport::cycles;
    let _: for<'a> fn(&'a OneNormReport<f64>) -> SpectralTermination = OneNormReport::termination;
    let _: for<'a> fn(&'a OneNormReport<f64>) -> f64 = OneNormReport::estimate;
    let _: for<'a> fn(&'a OneNormReport<f64>) -> usize = OneNormReport::applications;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> SpectralTermination = LowRankReport::termination;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> usize = LowRankReport::rank;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> f64 = LowRankReport::probe_residual;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> Option<f64> =
        LowRankReport::relative_probe_residual;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> usize = LowRankReport::completed_probes;
    let _: for<'a> fn(&'a LowRankReport<f64>) -> usize = LowRankReport::applications;
    let _: for<'a> fn(&'a RandomizedSvdBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        RandomizedSvdBuffer::u;
    let _: for<'a> fn(&'a RandomizedSvdBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        RandomizedSvdBuffer::vt;
    let _: for<'a> fn(&'a RandomizedSvdBuffer<f64>) -> &[f64] =
        RandomizedSvdBuffer::singular_values;
    let _: for<'a> fn(&'a InterpolativeBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        InterpolativeBuffer::columns;
    let _: for<'a> fn(&'a InterpolativeBuffer<f64>) -> scioxide_linalg::MatrixView<'a, f64> =
        InterpolativeBuffer::coefficients;
    let _: for<'a> fn(&'a InterpolativeBuffer<f64>) -> &[usize] =
        InterpolativeBuffer::column_indices;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> LinearSolveTermination =
        BlockKrylovReport::termination;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize = BlockKrylovReport::iterations;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> Option<f64> = BlockKrylovReport::max_residual;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize = BlockKrylovReport::restarts;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize = BlockKrylovReport::deflated;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize = BlockKrylovReport::active_rank;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize =
        BlockKrylovReport::generated_directions;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> usize = BlockKrylovReport::retained;
    let _: for<'a> fn(&'a BlockKrylovReport<f64>) -> bool = BlockKrylovReport::is_success;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> &[f64] = BlockKrylovResult::solution;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> usize = BlockKrylovResult::rows;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> usize = BlockKrylovResult::right_hand_sides;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> BlockKrylovReport<f64> =
        BlockKrylovResult::report;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> &[f64] = BlockKrylovResult::residuals;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> &[f64] = BlockKrylovResult::references;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> &[bool] = BlockKrylovResult::converged;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> Option<&[f64]> = BlockKrylovResult::history;
    let _: for<'a> fn(&'a BlockKrylovResult<f64>) -> bool = BlockKrylovResult::is_success;
    let _: for<'a> fn(&'a IterativeAccepted<'static, f64>) -> &'static [f64] =
        IterativeAccepted::solution;
    let _: for<'a> fn(&'a IterativeAccepted<'static, f64>) -> Option<&'static [f64]> =
        IterativeAccepted::history;
    let _: for<'a> fn(&'a IterativeAccepted<'static, f64>) -> Option<KrylovColumns<'static, f64>> =
        IterativeAccepted::columns;
    let _: for<'a> fn(&'a IterativeAccepted<'static, f64>) -> IterativeProgress<f64> =
        IterativeAccepted::progress;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> usize =
        BorderedSystemDiagnostics::nullity;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> f64 =
        BorderedSystemDiagnostics::compatibility_defect;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> f64 =
        BorderedSystemDiagnostics::projection_norm;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> f64 =
        BorderedSystemDiagnostics::original_residual_norm;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> f64 =
        BorderedSystemDiagnostics::projected_residual_norm;
    let _: for<'a> fn(&'a BorderedSystemDiagnostics<f64>) -> f64 =
        BorderedSystemDiagnostics::gauge_defect;
    let _: for<'a> fn(&'a KrylovDiagnostics<f64>) -> Option<ResidualReport<f64>> =
        KrylovDiagnostics::normal_residual;
    let _: for<'a> fn(
        &'a KrylovDiagnostics<f64>,
    ) -> Option<scioxide_linalg::ConditionEstimate<f64>> = KrylovDiagnostics::condition_estimate;
    let _: for<'a> fn(&'a KrylovDiagnostics<f64>) -> bool = KrylovDiagnostics::rank_loss_detected;
    let _: for<'a> fn(&'a KrylovDiagnostics<f64>) -> usize = KrylovDiagnostics::restarts;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> LinearSolveTermination =
        IterativeSolveReport::termination;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> Option<KrylovDiagnostics<f64>> =
        IterativeSolveReport::krylov;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> usize = IterativeSolveReport::iterations;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> Option<ResidualReport<f64>> =
        IterativeSolveReport::true_residual;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> PartialResultTrust =
        IterativeSolveReport::trust;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> bool = IterativeSolveReport::is_success;
    let _: for<'a> fn(&'a IterativeSolveReport<f64>) -> Option<f64> =
        IterativeSolveReport::diagnostic_residual_norm;
    let _: for<'a> fn(&'a IterativeSolveResult<f64>) -> scioxide_core::AlgorithmId =
        IterativeSolveResult::algorithm;
    let _: for<'a> fn(&'a IterativeSolveResult<f64>) -> &Vector<f64> =
        IterativeSolveResult::solution;
    let _: for<'a> fn(&'a IterativeSolveResult<f64>) -> IterativeSolveReport<f64> =
        IterativeSolveResult::report;
    let _: for<'a> fn(&'a IterativeSolveResult<f64>) -> Option<&[f64]> =
        IterativeSolveResult::history;
    let _: for<'a> fn(&'a IterativeSolveResult<f64>) -> bool = IterativeSolveResult::is_success;
    let _: fn(StructuredSolveReport<f64>) -> DirectSolveTermination =
        StructuredSolveReport::termination;
    let _: fn(CompositeSolveReport<f64>) -> DirectSolveTermination =
        CompositeSolveReport::termination;
    let _: fn(CompositeSolveReport<f64>) -> f64 = CompositeSolveReport::backward_error;
    let _: fn(StructuredSolveReport<f64>) -> f64 = StructuredSolveReport::backward_error;
    let _: fn(StructuredSolveReport<f64>) -> usize = StructuredSolveReport::levinson_solves;
    let _: fn(StructuredSolveReport<f64>) -> usize = StructuredSolveReport::gko_solves;
    let _: fn(StructuredSolveReport<f64>) -> usize = StructuredSolveReport::dense_lu_solves;
    let _: fn(StructuredSolveReport<f64>) -> usize = StructuredSolveReport::circulant_solves;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> DirectSolveTermination =
        MatrixEquationReport::termination;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> f64 = MatrixEquationReport::residual;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> f64 = MatrixEquationReport::backward_error;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> Option<f64> =
        MatrixEquationReport::symmetry_defect;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> Option<f64> =
        MatrixEquationReport::separation;
    let _: for<'a> fn(&'a MatrixEquationReport<f64>) -> usize = MatrixEquationReport::work;
    let _: for<'a> fn(&'a MatrixEquationResult<f64>) -> Option<&Matrix<f64>> =
        MatrixEquationResult::solution;
    let _: for<'a> fn(&'a MatrixEquationResult<f64>) -> MatrixEquationReport<f64> =
        MatrixEquationResult::report;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> DirectSolveTermination =
        RiccatiReport::termination;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> f64 = RiccatiReport::residual;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> f64 = RiccatiReport::backward_error;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> f64 = RiccatiReport::symmetry_defect;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> f64 = RiccatiReport::subspace_reciprocal_condition;
    let _: for<'a> fn(&'a RiccatiReport<f64>) -> Option<f64> = RiccatiReport::stability_margin;
    let _: for<'a> fn(&'a RiccatiResult<f64>) -> Option<&Matrix<f64>> = RiccatiResult::solution;
    let _: for<'a> fn(&'a RiccatiResult<f64>) -> Option<&Matrix<f64>> = RiccatiResult::gain;
    let _: for<'a> fn(&'a RiccatiResult<f64>) -> RiccatiReport<f64> = RiccatiResult::report;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> usize =
        SparseMaterializationReport::structural_fill;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> usize =
        SparseMaterializationReport::nonzero_fill;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> usize =
        SparseMaterializationReport::peak_intermediate_fill;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> usize =
        SparseMaterializationReport::admitted_workspace_bytes;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> usize =
        SparseMaterializationReport::work;
    let _: for<'a> fn(&'a SparseMaterializationReport<f64>) -> Option<f64> =
        SparseMaterializationReport::estimated_error;
    let _: for<'a> fn(&'a SparseMaterialization<f64>) -> &CsrMatrix<f64> =
        SparseMaterialization::value;
    let _: for<'a> fn(&'a SparseMaterialization<f64>) -> SparseMaterializationReport<f64> =
        SparseMaterialization::report;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> RefinementTermination =
        RefinementReport::termination;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> usize = RefinementReport::iterations;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> Option<BackwardError<f64>> =
        RefinementReport::initial_error;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> Option<BackwardError<f64>> =
        RefinementReport::final_error;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> Option<RefinementFallbackReason> =
        RefinementReport::fallback_reason;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> usize = RefinementReport::fallback_iterations;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> ScalarKind = RefinementReport::factor_precision;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> ScalarKind =
        RefinementReport::residual_precision;
    let _: for<'a> fn(&'a RefinementReport<f64>) -> bool = RefinementReport::is_success;
    let _: for<'a> fn(&'a GeneralizedSchurReport) -> scioxide_core::SolveCounts =
        GeneralizedSchurReport::counts;
    let _: fn(GeneralizedSchurReport) -> usize = GeneralizedSchurReport::iterations;
    let _: fn(GeneralizedSchurReport) -> usize = GeneralizedSchurReport::infinite_eigenvalues;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> usize =
        GeneralizedSchurReorderReport::selected_dimension;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> usize = GeneralizedSchurReorderReport::swaps;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> usize =
        GeneralizedSchurReorderReport::iterations;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> Option<ConditionEstimate<f64>> =
        GeneralizedSchurReorderReport::left_condition;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> Option<ConditionEstimate<f64>> =
        GeneralizedSchurReorderReport::right_condition;
    let _: fn(GeneralizedSchurReorderReport<f64>) -> Option<[ReliableEstimate<f64>; 2]> =
        GeneralizedSchurReorderReport::separation;
    let _: for<'a> fn(&'a HermitianEigenvaluesBuffer<f64>) -> &[f64] =
        HermitianEigenvaluesBuffer::eigenvalues;
    let _: for<'a> fn(
        &'a HermitianEigenvaluesBuffer<f64>,
    ) -> Option<&scioxide_linalg::HermitianProjectionReport<f64>> =
        HermitianEigenvaluesBuffer::projection_report;
    let _: for<'a> fn(&'a HermitianEigenvaluesResult<f64>) -> &[f64] =
        HermitianEigenvaluesResult::eigenvalues;
    let _: for<'a> fn(&'a HermitianEigenvaluesResult<f64>) -> scioxide_linalg::EigenReport =
        HermitianEigenvaluesResult::report;
    let _: for<'a> fn(
        &'a HermitianEigenvaluesResult<f64>,
    ) -> Option<&scioxide_linalg::HermitianProjectionReport<f64>> =
        HermitianEigenvaluesResult::projection_report;
    let _: for<'a> fn(
        &'a GeneralizedHermitianEigenvaluesBuffer<f64>,
    ) -> Option<&HermitianProjectionReport<f64>> =
        GeneralizedHermitianEigenvaluesBuffer::a_projection_report;
    let _: for<'a> fn(
        &'a GeneralizedHermitianEigenvaluesBuffer<f64>,
    ) -> Option<&HermitianProjectionReport<f64>> =
        GeneralizedHermitianEigenvaluesBuffer::b_projection_report;
    let _: for<'a> fn(&'a GeneralizedHermitianEigenvaluesBuffer<f64>) -> &[f64] =
        GeneralizedHermitianEigenvaluesBuffer::eigenvalues;
    let _: for<'a> fn(&'a GeneralizedHermitianEigenvaluesResult<f64>) -> &[f64] =
        GeneralizedHermitianEigenvaluesResult::eigenvalues;
    let _: for<'a> fn(
        &'a GeneralizedHermitianEigenvaluesResult<f64>,
    ) -> scioxide_linalg::EigenReport = GeneralizedHermitianEigenvaluesResult::report;
    let _: for<'a> fn(
        &'a GeneralizedHermitianEigenvaluesResult<f64>,
    ) -> Option<&scioxide_linalg::HermitianProjectionReport<f64>> =
        GeneralizedHermitianEigenvaluesResult::a_projection_report;
    let _: for<'a> fn(
        &'a GeneralizedHermitianEigenvaluesResult<f64>,
    ) -> Option<&scioxide_linalg::HermitianProjectionReport<f64>> =
        GeneralizedHermitianEigenvaluesResult::b_projection_report;
    let _: for<'a> fn(&'a EigenReport) -> scioxide_core::SolveCounts = EigenReport::counts;
    let _: for<'a> fn(&'a EigenReport) -> usize = EigenReport::iterations;
    let _: fn(SchurReorderReport<f64>) -> usize = SchurReorderReport::selected_dimension;
    let _: fn(SchurReorderReport<f64>) -> usize = SchurReorderReport::swaps;
    let _: fn(SchurReorderReport<f64>) -> usize = SchurReorderReport::iterations;
    let _: fn(SchurReorderReport<f64>) -> Option<ConditionEstimate<f64>> =
        SchurReorderReport::cluster_condition;
    let _: fn(SchurReorderReport<f64>) -> Option<ReliableEstimate<f64>> =
        SchurReorderReport::separation;
    let _: for<'a> fn(&'a GeneralEigenvaluesBuffer<f64>) -> &[Complex<f64>] =
        GeneralEigenvaluesBuffer::eigenvalues;
    let _: for<'a> fn(&'a GeneralEigenvaluesResult<f64>) -> &[Complex<f64>] =
        GeneralEigenvaluesResult::eigenvalues;
    let _: for<'a> fn(&'a GeneralEigenvaluesResult<f64>) -> scioxide_linalg::EigenReport =
        GeneralEigenvaluesResult::report;
    let _: fn(RankReport<f64>) -> usize = RankReport::rank;
    let _: fn(RankReport<f64>) -> usize = RankReport::maximum_rank;
    let _: fn(RankReport<f64>) -> f64 = RankReport::threshold;
    let _: fn(RankReport<f64>) -> f64 = RankReport::scale;
    let _: fn(RankReport<f64>) -> bool = RankReport::is_full_rank;
    let _: fn(DirectSolveReport<f64>) -> DirectSolveTermination = DirectSolveReport::termination;
    let _: fn(DirectSolveReport<f64>) -> RankReport<f64> = DirectSolveReport::rank;
    let _: fn(DirectSolveReport<f64>) -> Option<ConditionEstimate<f64>> =
        DirectSolveReport::condition;
    let _: fn(DirectSolveReport<f64>) -> Option<ResidualReport<f64>> = DirectSolveReport::residual;
    let _: for<'a> fn(&'a DirectVectorResult) -> Option<&Vector<f64>> = DirectSolveResult::solution;
    let _: for<'a> fn(&'a DirectVectorResult) -> &DirectSolveReport<f64> =
        DirectSolveResult::report;
}
