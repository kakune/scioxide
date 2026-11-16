//! Partial eigenproblems by thick-restart Lanczos (Hermitian) and Krylov–Schur (general).
//!
//! General cycles take Ritz vectors and restart directions from one Schur
//! factor. Arnoldi residual estimates decide when to check residuals between
//! the first and final cycles; acceptance always uses residuals of the
//! original operator or pencil. Real Ritz vectors need one real product.
use super::workspace::{PartialSpectrumRequirements, ShiftLayout, SpectrumLayout};
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};
use scioxide_core::AlgebraicScalar;
mod shift;
use super::{ActionSlot, OperatorStructure, SpectralTermination, Work, run_action};
use crate::kernel::finite;
use crate::kernel::subspace as krylov;
use crate::workspace::SolveSlot;
use crate::{
    HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    LinalgWorkspace, Matrix, SchurDecomposition, SchurOptions, SchurReorderOptions,
    SchurWorkspaceRequest, VectorView,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AttemptRecord, Budget, Complex, LinearOperator, NumericScalar, RealScalar, SciOxideError,
    Tolerance,
};
use shift::{ShiftInvertBuffers, ShiftInvertProblem};
/// Which eigenvalues [`PartialSpectrumSolver`] computes, and their order in the result.
///
/// Ties are ordered by increasing imaginary part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumSelection {
    /// Largest `|λ|` first.
    LargestMagnitude,
    /// Smallest `|λ|` first. For eigenvalues in the interior of the spectrum,
    /// shift-invert with [`SpectrumOperator::ShiftInvert`] is recommended.
    SmallestMagnitude,
    /// Smallest real part first.
    SmallestReal,
    /// Largest real part first.
    LargestReal,
}
/// Operator whose eigenpairs [`PartialSpectrumSolver`] computes: `A` itself, or a shift-invert
/// form.
///
/// A reference to a sized operator converts into `Direct`.
#[derive(Clone, Copy)]
pub enum SpectrumOperator<'a, S: NumericScalar> {
    /// Eigenpairs of `A x = λ x`, computed from products with `A`.
    Direct(&'a dyn LinearOperator<S>),
    /// Eigenpairs of `A x = λ M x` near the shift `σ`, computed from products
    /// with `(A − σM)⁻¹ M`.
    ///
    /// The Krylov method sees the transformed eigenvalues `θ = 1/(λ − σ)`, so
    /// the selection applies to `θ`: `LargestMagnitude` finds the eigenvalues
    /// nearest `σ`. Each Ritz value is mapped back by `λ = σ + 1/θ`, and
    /// residuals are computed with `A` and `M` themselves. You supply the
    /// inverse, with a factorization and solver of your choice.
    ///
    /// The Krylov basis is orthonormal in the Euclidean inner product. For
    /// Hermitian `A` and `M` and real `σ`, `(A − σM)⁻¹ M` is self-adjoint in
    /// the `M`-inner product but Hermitian only when `M` commutes with `A`
    /// (for example `M = I`). Otherwise use [`OperatorStructure::General`]:
    /// `Hermitian` structure makes the solve fail with `InvalidInput`
    /// (`Inconsistent`).
    ShiftInvert {
        /// The operator `A`, used for the residuals.
        a: &'a dyn LinearOperator<S>,
        /// The operator `(A − σM)⁻¹`.
        inverse: &'a dyn LinearOperator<S>,
        /// The mass matrix `M`; `None` means the identity.
        mass: Option<&'a dyn LinearOperator<S>>,
        /// The shift `σ` that `inverse` uses.
        shift: S,
    },
}
impl<S: NumericScalar> core::fmt::Debug for SpectrumOperator<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Direct(_) => f.write_str("Direct"),
            Self::ShiftInvert { mass, shift, .. } => f
                .debug_struct("ShiftInvert")
                .field("mass", &mass.is_some())
                .field("shift", shift)
                .finish_non_exhaustive(),
        }
    }
}
impl<'a, S: NumericScalar, T: LinearOperator<S>> From<&'a T> for SpectrumOperator<'a, S> {
    fn from(operator: &'a T) -> Self {
        Self::Direct(operator)
    }
}
/// Solver for a few eigenpairs of a large operator by restarted Krylov methods.
///
/// The operator is `n × n` and used only through products; see
/// [`SpectrumOperator`]. Each cycle extends an Arnoldi basis up to dimension
/// `m`, using classical Gram–Schmidt with one reorthogonalization when the norm
/// drops by more than a factor `1/√2` (Daniel, Gragg, Kaufman and Stewart,
/// 1976). It then computes the Ritz pairs of the projected matrix and restarts:
///
/// - General operators use the Krylov–Schur method (Stewart, 2001). The Schur
///   form of the projected matrix is reordered so that the wanted Ritz values
///   come first, and that invariant subspace is kept.
/// - Hermitian operators use thick-restart Lanczos (Wu and Simon, 2000). The
///   wanted Ritz vectors are kept together with their residual couplings.
///
/// A restart keeps `max(count, d/2)` vectors, but at most `d − 2`, where `d` is
/// the basis dimension reached. Pairs that already meet the tolerance are kept
/// first and checked again in later cycles. A cycle costs about `m − kept`
/// products, O(n m²) operations for orthogonalization and O(m³) for the
/// projected eigenproblem.
///
/// Ritz pairs are accepted only by residuals computed from fresh products with
/// the operator (for shift-invert, with `A` and `M`). These checks run in the
/// first and last cycles, when the basis becomes invariant, and in any cycle
/// where the Arnoldi estimate `β |eₘᵀ y|` of some wanted pair meets the
/// tolerance. For Hermitian operators each Ritz value is recomputed as the
/// Rayleigh quotient `xᴴ A x / xᴴ x` from the same product.
///
/// The value holds only the settings; `n` is read from the operator of each
/// call. Defaults: basis dimension 30, at most 100 000 operator applications,
/// residual tolerance `atol = rtol = √ε`, `LargestMagnitude` selection and
/// `General` structure. The iteration limit of the execution options bounds
/// the restart cycles (default 100). The dense eigenproblems of the projected
/// matrices use the other execution options but no iteration limit.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, PartialSpectrumSolver, SpectralTermination};
/// let a = Matrix::<f64>::identity(2)?;
/// let (output, report) =
///     PartialSpectrumSolver::new(1).solve(&a.view().try_finite()?, &[1.0, 0.0][..])?;
/// assert_eq!(report.termination(), SpectralTermination::Converged);
/// // The identity has the single eigenvalue 1.
/// assert!((output.eigenvalues()[0].re - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. W. Stewart, "A Krylov–Schur algorithm for large eigenproblems",
///   *SIAM J. Matrix Anal. Appl.* 23(3), 2001.
/// - K. Wu and H. Simon, "Thick-restart Lanczos method for large symmetric
///   eigenvalue problems", *SIAM J. Matrix Anal. Appl.* 22(2), 2000.
/// - J. W. Daniel, W. B. Gragg, L. Kaufman and G. W. Stewart,
///   "Reorthogonalization and stable algorithms for updating the Gram–Schmidt
///   QR factorization", *Math. Comp.* 30(136), 1976.
#[derive(Debug, Clone, Copy)]
pub struct PartialSpectrumSolver<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Requested number of Ritz pairs.
    count: usize,
    /// Maximum basis dimension; at least count+2 unless it spans the whole operator.
    basis: Budget,
    /// Total forward provider calls, including original-operator residual verification.
    applications: Budget,
    /// Independent residual acceptance `||Av-lambda v|| <= abs + rel ||Av||`.
    tolerance: scioxide_core::ResidualTolerance<R>,
    /// Ritz selection order.
    selection: SpectrumSelection,
    /// `Hermitian` asserts a Hermitian operator and uses the thick-restarted Lanczos specialization.
    structure: OperatorStructure,
}
impl<R: RealScalar> PartialSpectrumSolver<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the maximum Krylov basis dimension `m` (default 30).
    ///
    /// `m` must be at least `count + 2`, unless `m ≥ n`.
    pub const fn with_basis(mut self, basis: Budget) -> Self {
        self.basis = basis;
        self
    }

    /// Sets the maximum number of operator applications, including residual
    /// checks (default 100 000).
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the residual tolerance (default `atol = rtol = √ε`).
    ///
    /// A Ritz pair `(λ, x)` with `‖x‖₂ = 1` is accepted when
    /// `‖A x − λ x‖₂ ≤ atol + rtol ‖A x‖₂`. For shift-invert the test is
    /// `‖A x − λ M x‖₂ ≤ atol + rtol max(‖A x‖₂, |λ| ‖M x‖₂)`.
    pub const fn with_tolerance(mut self, tolerance: scioxide_core::ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets which eigenvalues to compute (default `LargestMagnitude`).
    pub const fn with_selection(mut self, selection: SpectrumSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Sets the operator structure (default `General`).
    ///
    /// `Hermitian` selects thick-restart Lanczos and real eigenvalues. The
    /// assumption is checked on each projected matrix.
    pub const fn with_structure(mut self, structure: OperatorStructure) -> Self {
        self.structure = structure;
        self
    }

    /// Returns the number of eigenpairs requested.
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns the maximum Krylov basis dimension.
    pub const fn basis(&self) -> Budget {
        self.basis
    }

    fn iteration_budget(&self) -> Budget {
        self.execution
            .iteration_budget()
            .unwrap_or(Budget::ONE.saturating_add(99))
    }

    /// Returns the maximum number of operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the residual tolerance.
    pub const fn tolerance(&self) -> scioxide_core::ResidualTolerance<R> {
        self.tolerance
    }

    /// Returns which eigenvalues are computed.
    pub const fn selection(&self) -> SpectrumSelection {
        self.selection
    }

    /// Returns the operator structure.
    pub const fn structure(&self) -> OperatorStructure {
        self.structure
    }
}
impl<R: RealScalar> PartialSpectrumSolver<R> {
    /// Returns default options for `count` eigenpairs.
    pub fn new(count: usize) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            count,
            basis: Budget::ONE.saturating_add(29),
            applications: Budget::ONE.saturating_add(99_999),
            tolerance: scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
            ),
            selection: SpectrumSelection::LargestMagnitude,
            structure: OperatorStructure::General,
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct SpectrumCall<R: RealScalar> {
    n: usize,
    options: PartialSpectrumSolver<R>,
}
/// Eigenpairs computed by [`PartialSpectrumSolver`]: Ritz values, Ritz vectors and residual norms.
///
/// Values and vectors are complex, also for real operators. They are in
/// selection order, and each vector has `‖x‖₂ = 1`. The residuals are
/// `‖A x − λ x‖₂`, or `‖A x − λ M x‖₂` for shift-invert, computed from fresh
/// products.
///
/// The buffer can hold fewer pairs than requested, and only `Converged`
/// guarantees that every pair meets the tolerance. After a limit or
/// cancellation, it holds the most recent pairs whose residuals were computed,
/// possibly none.
#[derive(Debug)]
pub struct PartialSpectrumBuffer<R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    values: Vec<Complex<R>>,
    vectors: Matrix<Complex<R>>,
    residuals: Vec<R>,
    count: usize,
}
impl<R: RealScalar> PartialSpectrumBuffer<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Allocates a buffer for `count` eigenpairs of `n × n` operators.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n · count` overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(n: usize, count: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            values: try_filled(
                count,
                Complex::ZERO,
                scioxide_core::quantity!("result.values"),
            )?,
            vectors: Matrix::zeros(n, count)?,
            residuals: try_filled(count, R::ZERO, scioxide_core::quantity!("result.residuals"))?,
            count: 0,
        })
    }
    /// Returns the Ritz values in selection order; there may be fewer than requested.
    pub fn eigenvalues(&self) -> &[Complex<R>] {
        &self.values[..self.count]
    }
    /// Returns the Ritz vectors as the columns of an `n × r` view, each with unit 2-norm.
    pub fn eigenvectors(&self) -> crate::MatrixView<'_, Complex<R>> {
        let mut view = self.vectors.view();
        view.cols = self.count;
        view
    }
    /// Returns the residual norms of the Ritz pairs.
    pub fn residuals(&self) -> &[R] {
        &self.residuals[..self.count]
    }
}
/// Workspace of [`PartialSpectrumSolver`].
///
/// It holds O(n m + m²) scalars for basis dimension `m`.
#[derive(Debug)]
pub struct PartialSpectrumWorkspace<S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    n: usize,
    m: usize,
    projection_roundoff: S::Real,
    basis: Vec<S>,
    retained: Vec<S>,
    image: Vec<S>,
    coefficients: Vec<S>,
    real_input: Vec<S>,
    real_image: Vec<S>,
    complex_image: Vec<Complex<S::Real>>,
    provider: Vec<S>,
    h: Matrix<S>,
    projected: Matrix<S>,
    solver: ProjectedSolver<S>,
    other_solver: Option<ProjectedSolver<S>>,
    order: Vec<usize>,
    mask: Vec<bool>,
    candidate: PartialSpectrumBuffer<S::Real>,
    proposal: PartialSpectrumBuffer<S::Real>,
    shift_invert: Option<ShiftInvertBuffers<S>>,
    stop: Option<SpectralTermination>,
    candidate_cycles: usize,
    proposal_cycles: usize,
    slot: SolveSlot<()>,
}
/// Selected projected algorithm. A workspace keeps the other variant after a switch.
// Embedded arenas keep workspace construction fallible and avoid an extra
// allocation solely to shrink the enum; neither variant lives in a hot array.
#[derive(Debug)]
enum ProjectedSolver<S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    Hermitian {
        eigen: HermitianEigenDecomposition<S>,
        work: LinalgWorkspace<'static, S>,
    },
    General {
        vectors: Matrix<Complex<S::Real>>,
        vector_work: Vec<S>,
        phases: Vec<Complex<S::Real>>,
        schur: SchurDecomposition<S>,
        schur_work: LinalgWorkspace<'static, S>,
    },
}
impl<S: NumericScalar> ProjectedSolver<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn empty(structure: OperatorStructure) -> Self {
        match structure {
            OperatorStructure::Hermitian => Self::Hermitian {
                eigen: HermitianEigenDecomposition::empty(),
                work: LinalgWorkspace::new(),
            },
            OperatorStructure::General => Self::General {
                vectors: Matrix::empty(),
                vector_work: Vec::new(),
                phases: Vec::new(),
                schur: SchurDecomposition::empty(),
                schur_work: LinalgWorkspace::new(),
            },
        }
    }
    fn structure(&self) -> OperatorStructure {
        match self {
            Self::Hermitian { .. } => OperatorStructure::Hermitian,
            Self::General { .. } => OperatorStructure::General,
        }
    }
    fn storage_plan(
        &self,
        plan: &mut StoragePlan,
        m: usize,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Hermitian { eigen, work } => {
                eigen.action_storage(plan, m)?;
                plan.dense(work, &requirements.storage.dense[0])?;
            }
            Self::General {
                vectors,
                vector_work,
                phases,
                schur,
                schur_work,
            } => {
                plan.matrix(vectors, m, m)?;
                plan.buffer(vector_work, sum([product(&[4, m, m])?, product(&[2, m])?])?)?;
                plan.buffer(phases, m)?;
                schur.action_storage(plan, m)?;
                plan.dense(schur_work, &requirements.storage.dense[0])?;
            }
        }
        Ok(())
    }
    fn configure(
        &mut self,
        m: usize,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Hermitian { eigen, work } => {
                eigen.reserve(m, m)?;
                work.try_reserve(&requirements.storage.dense[0])?;
            }
            Self::General {
                vectors,
                vector_work,
                phases,
                schur,
                schur_work,
            } => {
                vectors.try_resize_flat(m, m)?;
                resize(
                    vector_work,
                    sum([product(&[4, m, m])?, product(&[2, m])?])?,
                    S::ZERO,
                )?;
                resize(phases, m, Complex::ZERO)?;
                schur.reserve(m)?;
                schur_work.try_reserve(&requirements.storage.dense[0])?;
            }
        }
        Ok(())
    }
    fn value(&self, index: usize) -> Complex<S::Real> {
        match self {
            Self::Hermitian { eigen, .. } => Complex::from_real(eigen.eigenvalues()[index]),
            Self::General { schur, .. } => schur.eigenvalues()[index],
        }
    }
    fn vector(&self, row: usize, column: usize) -> Complex<S::Real> {
        match self {
            Self::Hermitian { eigen, .. } => {
                crate::eigen::into_complex(eigen.eigenvectors().get_element(row, column))
            }
            Self::General { vectors, .. } => vectors.get_element(row, column),
        }
    }
}
impl<S: NumericScalar> ActionSlot for PartialSpectrumWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> PartialSpectrumWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Returns the attempt record of the most recent solve, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
impl<R: RealScalar> SpectrumCall<R> {
    fn workspace_requirements<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
    ) -> Result<PartialSpectrumRequirements<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let (provider, shift) = match op.into() {
            SpectrumOperator::Direct(op) => {
                crate::checked::expect_shape([self.n, self.n], [op.rows(), op.cols()])?;
                (op.apply_workspace_len()?, None)
            }
            SpectrumOperator::ShiftInvert {
                a, inverse, mass, ..
            } => {
                crate::checked::expect_shape([self.n, self.n], [a.rows(), a.cols()])?;
                crate::checked::expect_shape([self.n, self.n], [inverse.rows(), inverse.cols()])?;
                let shifted = super::ShiftInvertOperator::new(inverse, mass)?;
                (
                    shifted.apply_workspace_len()?,
                    Some(ShiftLayout {
                        a: a.apply_workspace_len()?,
                        mass: mass.map_or(Ok(0), |m| m.apply_workspace_len())?,
                    }),
                )
            }
        };
        let n = self.n;
        let m = self.options.basis.get().min(n);
        let k = self.options.count;
        let mut dense = [crate::LinalgRequirements::default(); 2];
        let complex = dense;
        let mut payload = sum([
            bytes::<S>(sum([
                product(&[m, m])?,
                product(&[n, sum([product(&[2, m])?, 4])?])?,
                product(&[2, m, m])?,
                m,
                provider,
            ])?)?,
            bytes::<Complex<R>>(sum([n, product(&[2, k, sum([n, 1])?])?])?)?,
            bytes::<R>(product(&[2, k])?)?,
            bytes::<usize>(m)?,
            bytes::<bool>(m)?,
        ])?;
        match self.options.structure {
            OperatorStructure::Hermitian => {
                dense[0] = HermitianEigenDecomposition::<S>::workspace_requirements(
                    &HermitianEigenWorkspaceRequest::new(
                        m,
                        &HermitianEigenOptions::<R>::default().with_execution_options(
                            super::projected_execution_options(self.options.execution),
                        ),
                    ),
                )?;
                payload = sum([payload, bytes::<S>(product(&[m, m])?)?, bytes::<R>(m)?])?;
            }
            OperatorStructure::General => {
                dense[0] =
                    SchurDecomposition::<S>::workspace_requirements(&SchurWorkspaceRequest::new(
                        m,
                        &SchurOptions::<R>::default().with_execution_options(
                            super::projected_execution_options(self.options.execution),
                        ),
                    ))?
                    .max(&SchurDecomposition::<S>::workspace_requirements(
                        &SchurWorkspaceRequest::reorder(
                            m,
                            &SchurReorderOptions::default().with_execution_options(
                                super::projected_execution_options(self.options.execution),
                            ),
                        ),
                    )?);
                payload = sum([
                    payload,
                    bytes::<S>(sum([product(&[6, m, m])?, product(&[2, m])?])?)?,
                    bytes::<Complex<R>>(sum([product(&[m, m])?, product(&[2, m])?])?)?,
                    bytes::<crate::SchurBlock>(m)?,
                ])?;
            }
        }
        dense[0] = dense[0].max(&krylov::block_requirements::<S>(
            n,
            m,
            m,
            &self.options.execution,
        )?);
        if let Some(shift) = shift {
            payload = sum([
                payload,
                bytes::<S>(sum([product(&[2, n])?, shift.a, shift.mass])?)?,
                bytes::<Complex<R>>(product(&[2, n])?)?,
            ])?;
        }
        PartialSpectrumRequirements::new(
            SpectrumLayout {
                n,
                basis: m,
                count: k,
                structure: self.options.structure,
                provider,
                shift,
            },
            self.options.execution,
            payload,
            dense,
            complex,
        )
    }
    fn workspace<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
    ) -> Result<PartialSpectrumWorkspace<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let requirements = self.workspace_requirements(op)?;
        let mut w = PartialSpectrumWorkspace::new();
        w.try_reserve(&requirements)?;
        Ok(w)
    }

    fn solve<'o, 'i, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
        initial: impl Into<VectorView<'i, S>>,
    ) -> Result<(PartialSpectrumBuffer<R>, PartialSpectrumReport), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        let mut workspace = self.workspace(op)?;
        let mut output = PartialSpectrumBuffer::new(self.n, self.options.count)?;
        let report = self.solve_into(
            op,
            initial,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn solve_into<'o, 'i, 's, 'c, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
        initial: impl Into<VectorView<'i, S>>,
        result: &mut PartialSpectrumBuffer<R>,
        w: &mut PartialSpectrumWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<PartialSpectrumReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        let initial = initial.into();
        w.slot.reject();
        let requirements = self.workspace_requirements(op)?;
        crate::checked::expect_shape([self.n], [initial.len()])?;
        crate::checked::expect_shape([self.n], [result.vectors.rows()])?;
        crate::checked::expect_shape([self.options.count], [result.values.len()])?;
        let _ = initial.try_finite()?;
        if self.options.count > 0 && initial.iter().all(|x| x == S::ZERO) {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("initial"),
                "starting vector must be nonzero",
            ));
        }
        if let SpectrumOperator::ShiftInvert { shift, .. } = op {
            crate::checked::validate_input(shift, "shift")?;
        }
        w.prepare(&requirements)?;
        run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "PartialSpectrum", 1),
            self.options.applications,
            &self.options.execution,
            |w, mut work| match op {
                SpectrumOperator::Direct(a) => self.run(a, initial, result, w, &mut work, None),
                SpectrumOperator::ShiftInvert {
                    a,
                    inverse,
                    mass,
                    shift,
                } => self.shift_run(
                    shift::ShiftedProblem {
                        a,
                        inverse,
                        mass,
                        shift,
                    },
                    initial,
                    result,
                    w,
                    &mut work,
                ),
            },
            |report: &PartialSpectrumReport| report.termination() == SpectralTermination::Converged,
        )
    }

    fn run<S: NumericScalar<Real = R>>(
        &self,
        op: &dyn LinearOperator<S>,
        initial: VectorView<'_, S>,
        result: &mut PartialSpectrumBuffer<R>,
        w: &mut PartialSpectrumWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        mut shift: Option<(&ShiftInvertProblem<'_, S>, &mut ShiftInvertBuffers<S>)>,
    ) -> Result<PartialSpectrumReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let execution = work.execution;

        let n = self.n;
        let wanted = self.options.count;
        if w.provider.len() != op.apply_workspace_len()? {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("op"),
                "operator changed its scratch requirement after preparation",
            ));
        }

        w.projection_roundoff = R::ZERO;
        w.candidate.count = 0;
        w.proposal.count = 0;
        w.candidate_cycles = 0;
        w.stop = None;
        let outcome = (|| {
            execution.check()?;
            if wanted == 0 {
                w.stop = Some(SpectralTermination::Converged);
            } else {
                // Preflight rejected a zero start, and the norm bounds its largest entry.
                let norm = crate::reduction::l2_norm_with_context(initial, work.kernels)?;
                for (v, x) in w.basis[..n].iter_mut().zip(initial.iter()) {
                    *v = x / S::from_real(norm);
                }
                w.h.as_mut_slice().fill(S::ZERO);
                let mut start = 0usize;
                'cycles: for cycle in 0..self.options.iteration_budget().get() {
                    if !work.iteration()? {
                        w.stop = Some(SpectralTermination::EvaluationLimit);
                        break 'cycles;
                    }
                    let core::ops::ControlFlow::Continue((dimension, happy)) =
                        self.expand_basis(op, w, work, start)?
                    else {
                        break 'cycles;
                    };
                    w.proposal_cycles = cycle + 1;
                    let core::ops::ControlFlow::Continue((count, converged)) = self
                        .verify_ritz_pairs(
                            op,
                            w,
                            work,
                            dimension,
                            shift
                                .as_mut()
                                .map(|(problem, buffers)| (*problem, &mut **buffers)),
                            cycle == 0
                                || cycle + 1 == self.options.iteration_budget().get()
                                || happy
                                || dimension == n,
                        )?
                    else {
                        break 'cycles;
                    };
                    if count > 0 {
                        w.proposal.count = count;
                        core::mem::swap(&mut w.candidate, &mut w.proposal);
                        core::mem::swap(&mut w.candidate_cycles, &mut w.proposal_cycles);
                        w.proposal.count = 0;
                    }
                    if converged {
                        w.stop = Some(SpectralTermination::Converged);
                        break;
                    }
                    if happy || dimension == n {
                        w.stop = Some(SpectralTermination::InvariantSubspace);
                        break;
                    }
                    w.stop = Some(SpectralTermination::IterationLimit);
                    start = self.restart_basis(w, work, dimension)?;
                }
            }
            execution.check()
        })();
        let termination = super::action_termination(
            outcome,
            w.stop.unwrap_or(SpectralTermination::EvaluationLimit),
        )?;
        if w.proposal.count > 0 && w.proposal.count >= w.candidate.count {
            core::mem::swap(&mut w.candidate, &mut w.proposal);
            core::mem::swap(&mut w.candidate_cycles, &mut w.proposal_cycles);
        }
        result.count = w.candidate.count;
        super::publish_snapshot(
            w.candidate.count,
            &w.candidate.values,
            &w.candidate.residuals,
            &mut result.values,
            &mut result.residuals,
            [(&w.candidate.vectors, &mut result.vectors)],
            (Complex::ZERO, R::ZERO),
        );
        Ok(PartialSpectrumReport {
            termination,
            applications: work.count,
            cycles: w.candidate_cycles,
        })
    }
    fn expand_basis<S: NumericScalar<Real = R>>(
        &self,
        op: &dyn LinearOperator<S>,
        w: &mut PartialSpectrumWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        start: usize,
    ) -> Result<core::ops::ControlFlow<(), (usize, bool)>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let n = self.n;
        let m = self.options.basis.get().min(n);

        let mut dimension = start;
        let mut happy = false;
        for column in start..m {
            work.execution.check()?;
            if !work.apply(
                op,
                &w.basis[column * n..(column + 1) * n],
                &mut w.image,
                &mut w.provider,
            )? {
                w.stop = Some(SpectralTermination::EvaluationLimit);
                return Ok(core::ops::ControlFlow::Break(()));
            }
            let before = finite::norm(&w.image)?;
            if self.options.structure == OperatorStructure::Hermitian {
                w.projection_roundoff = w.projection_roundoff.max(
                    R::try_from_usize_exact(8)?
                        * R::EPSILON
                        * crate::checked::rounded_count::<R>(n.max(1))?
                        * finite::maximum(&w.basis[..(column + 1) * n], work.kernels.dispatch())?
                        * finite::maximum(&w.image, work.kernels.dispatch())?,
                );
                // Retained Ritz values form a diagonal block. Only its
                // residual row couples to the first new Lanczos direction;
                // subsequent directions have the three-term recurrence.
                for j in 0..column {
                    let coupling = w.h.get_element(column, j).conj();
                    w.h.as_mut_slice()[j * m + column] = coupling;
                    finite::axpy(
                        &mut w.image,
                        -coupling,
                        &w.basis[j * n..(j + 1) * n],
                        work.kernels.dispatch(),
                    )?;
                }
            }
            let beta = krylov::project(
                &mut w.image,
                &w.basis[..(column + 1) * n],
                &mut w.coefficients,
                |j, c| w.h.as_mut_slice()[j * m + column] = w.h.get_element(j, column) + c,
                work.kernels,
            )?;
            w.h.as_mut_slice()[(column + 1) * m + column] = S::from_real(beta);
            dimension = column + 1;
            if beta <= crate::checked::dimension_epsilon::<R>(n.max(1))? * before {
                happy = true;
                break;
            }
            for (v, &x) in w.basis[(column + 1) * n..(column + 2) * n]
                .iter_mut()
                .zip(&w.image)
            {
                *v = x / S::from_real(beta);
            }
        }
        Ok(core::ops::ControlFlow::Continue((dimension, happy)))
    }
    fn verify_ritz_pairs<S: NumericScalar<Real = R>>(
        &self,
        op: &dyn LinearOperator<S>,
        w: &mut PartialSpectrumWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        dimension: usize,
        mut shift: Option<(&ShiftInvertProblem<'_, S>, &mut ShiftInvertBuffers<S>)>,
        force_verification: bool,
    ) -> Result<core::ops::ControlFlow<(), (usize, bool)>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let n = self.n;
        let wanted = self.options.count;

        w.projected.try_resize_flat(dimension, dimension)?;
        for i in 0..dimension {
            for j in 0..dimension {
                w.projected.as_mut_slice()[i * dimension + j] = w.h.get_element(i, j);
            }
        }
        if self.options.structure == OperatorStructure::Hermitian {
            let threshold = self.options.tolerance.threshold(
                w.projected
                    .as_slice()
                    .iter()
                    .fold(R::ZERO, |v, x| v.max(x.abs())),
            )?;
            for i in 0..dimension {
                for j in i..dimension {
                    let a = w.projected.get_element(i, j);
                    let b = w.projected.get_element(j, i).conj();
                    if (a - b).abs() > threshold.max(w.projection_roundoff) {
                        return Err(crate::checked::inconsistent(
                            scioxide_core::argument!("op"),
                            "projected operator contradicts the Hermitian assumption",
                        ));
                    }
                    let v = (a * S::from_real(R::HALF)) + (b * S::from_real(R::HALF));
                    w.projected.as_mut_slice()[i * dimension + j] = v;
                    w.projected.as_mut_slice()[j * dimension + i] = v.conj();
                }
            }
        }
        work.execution.check()?;
        match &mut w.solver {
            ProjectedSolver::Hermitian {
                eigen,
                work: scratch,
            } => {
                HermitianEigenDecomposition::decompose_into(
                    &w.projected,
                    crate::HermitianInput::Exact,
                    &HermitianEigenOptions::default().with_execution_options(
                        super::projected_execution_options(self.options.execution),
                    ),
                    eigen,
                    scratch,
                    work.kernels.scoped(&mut *work.solve),
                )?;
            }
            ProjectedSolver::General {
                vectors,
                vector_work,
                phases,
                schur,
                schur_work,
            } => {
                SchurDecomposition::decompose_into(
                    w.projected.view(),
                    &SchurOptions::default().with_execution_options(
                        super::projected_execution_options(self.options.execution),
                    ),
                    schur,
                    schur_work,
                    work.kernels.scoped(&mut *work.solve),
                )?;
                crate::eigen::right_vectors_from_schur(
                    schur,
                    vectors,
                    vector_work,
                    phases,
                    work.kernels,
                )?;
            }
        }
        for (i, v) in w.order[..dimension].iter_mut().enumerate() {
            *v = i;
        }
        w.order[..dimension].sort_unstable_by(|&a, &b| {
            compare(w.solver.value(a), w.solver.value(b), self.options.selection)
        });
        w.mask[..dimension].fill(false);
        let count = wanted.min(dimension);
        let mut converged = count == wanted;
        if !force_verification {
            let beta = w.h.get_element(dimension, dimension - 1).abs();
            let mut ready = false;
            for &index in &w.order[..count] {
                let estimate = beta * w.solver.vector(dimension - 1, index).norm();
                ready |= estimate
                    <= self
                        .options
                        .tolerance
                        .threshold(w.solver.value(index).norm())?;
            }
            if !ready {
                // No target is ready to lock. Retain the previous independently
                // verified snapshot until a target, stop or final cycle needs
                // fresh original-operator residuals.
                return Ok(core::ops::ControlFlow::Continue((0, false)));
            }
        }
        if let ProjectedSolver::Hermitian {
            eigen,
            work: scratch,
        } = &mut w.solver
        {
            krylov::combine_block(
                crate::MatrixViewMut::from_slice(count, n, &mut w.retained[..count * n])?
                    .transpose(),
                &w.basis[..dimension * n],
                |j, target| eigen.eigenvectors().get_element(j, w.order[target]),
                &mut w.coefficients,
                scratch,
                self.options.execution.execution_policy(),
                work.kernels,
            )?;
        }
        for target in 0..count {
            let index = w.order[target];
            let lambda = w.solver.value(index);
            if let ProjectedSolver::Hermitian { .. } = &w.solver {
                for (row, &value) in w.retained[target * n..(target + 1) * n].iter().enumerate() {
                    w.proposal.vectors.as_mut_slice()[row * wanted + target] =
                        crate::eigen::into_complex(value);
                }
            } else {
                krylov::combine(
                    &mut w.complex_image,
                    &w.basis[..dimension * n],
                    (0..dimension).map(|j| w.solver.vector(j, index)),
                    crate::eigen::into_complex,
                    work.kernels.dispatch(),
                )?;
                for (row, &value) in w.complex_image.iter().enumerate() {
                    w.proposal.vectors.as_mut_slice()[row * wanted + target] = value;
                }
            }
            let norm = crate::reduction::l2_norm_with_context(
                w.proposal.vectors.view().col(target)?,
                work.kernels,
            )?;
            if norm == R::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "projected Ritz vector vanished",
                ));
            }
            for row in 0..n {
                w.proposal.vectors.as_mut_slice()[row * wanted + target] =
                    w.proposal.vectors.get_element(row, target) / Complex::from_real(norm);
            }
            let (lambda, residual, anorm) = if let Some((problem, buffers)) = &mut shift {
                let Some(verified) =
                    problem.verify(lambda, &w.proposal.vectors, target, buffers, work)?
                else {
                    w.stop = Some(SpectralTermination::EvaluationLimit);
                    return Ok(core::ops::ControlFlow::Break(()));
                };
                (verified.value, verified.residual, verified.scale)
            } else {
                if !shift::complex_apply(
                    op,
                    &w.proposal.vectors,
                    target,
                    &mut w.complex_image,
                    shift::OperatorScratch {
                        input: &mut w.real_input,
                        image: &mut w.real_image,
                        scratch: &mut w.provider,
                    },
                    work,
                )? {
                    w.stop = Some(SpectralTermination::EvaluationLimit);
                    return Ok(core::ops::ControlFlow::Break(()));
                }
                // Recompute the Hermitian Rayleigh quotient using the same original
                // operator image as the residual. Projected eigenvalue roundoff must
                // not impose the small eigensolver's accuracy on a tighter action.
                let lambda = if self.options.structure == OperatorStructure::Hermitian {
                    let vector = w.proposal.vectors.view().col(target)?;
                    let numerator = crate::reduction::dot_with_context(
                        vector,
                        VectorView::from_slice(&w.complex_image),
                        true,
                        work.kernels,
                    )?;
                    let denominator =
                        crate::reduction::dot_with_context(vector, vector, true, work.kernels)?;
                    Complex::from_real(numerator.re / denominator.re)
                } else {
                    lambda
                };
                let anorm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(&w.complex_image),
                    work.kernels,
                )?;
                for i in 0..n {
                    w.complex_image[i] =
                        w.complex_image[i] - lambda * w.proposal.vectors.get_element(i, target);
                }
                let residual = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(&w.complex_image),
                    work.kernels,
                )?;
                (lambda, residual, anorm)
            };
            w.proposal.values[target] = lambda;
            w.proposal.residuals[target] = residual;
            w.proposal.count = target + 1;
            w.mask[index] = residual <= self.options.tolerance.threshold(anorm)?;
            converged &= w.mask[index];
        }
        Ok(core::ops::ControlFlow::Continue((count, converged)))
    }
    fn restart_basis<S: NumericScalar<Real = R>>(
        &self,
        w: &mut PartialSpectrumWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        dimension: usize,
    ) -> Result<usize, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let n = self.n;
        let m = self.options.basis.get().min(n);
        let wanted = self.options.count;
        match &mut w.solver {
            ProjectedSolver::Hermitian {
                eigen,
                work: scratch,
            } => {
                let keep = wanted.max(dimension / 2).min(dimension - 2);
                let beta = w.h.get_element(dimension, dimension - 1);
                // Put independently converged targets first in the retained block;
                // projection always removes their components before active ones.
                w.order[..keep].sort_unstable_by_key(|&index| (!w.mask[index], index));
                // Soft locking retains all wanted directions and guard Ritz
                // vectors. Original residuals are rechecked for acceptance, so
                // a deteriorated target becomes active again without deflation.
                krylov::combine_block(
                    crate::MatrixViewMut::from_slice(keep, n, &mut w.retained[..keep * n])?
                        .transpose(),
                    &w.basis[..dimension * n],
                    |j, target| eigen.eigenvectors().get_element(j, w.order[target]),
                    &mut w.coefficients,
                    scratch,
                    self.options.execution.execution_policy(),
                    work.kernels,
                )?;
                w.basis
                    .copy_within(dimension * n..(dimension + 1) * n, keep * n);
                w.basis[..keep * n].copy_from_slice(&w.retained[..keep * n]);
                w.h.as_mut_slice().fill(S::ZERO);
                for j in 0..keep {
                    let index = w.order[j];
                    w.h.as_mut_slice()[j * m + j] = S::from_real(eigen.eigenvalues()[index]);
                    w.h.as_mut_slice()[keep * m + j] =
                        beta * eigen.eigenvectors().get_element(dimension - 1, index);
                }
                Ok(keep)
            }
            ProjectedSolver::General {
                schur, schur_work, ..
            } => {
                // Krylov-Schur restart: retain selected invariant directions and the residual coupling row.
                // (Stewart's Krylov-Schur method.) With `H = Z T Z^H` reordered so
                // the wanted Ritz values lead, `V Z[:, :k]` spans the retained
                // subspace, `T[:k, :k]` is its projected operator, and the old
                // residual term `beta v_{m+1} e_m^T` becomes the coupling row
                // `beta Z[m-1, :k]`. The last basis vector is carried over and
                // the Arnoldi process continues from column k. Keeping at least
                // half the basis (but leaving two free columns) balances restart
                // cost against convergence.
                for (i, v) in w.order[..dimension].iter_mut().enumerate() {
                    *v = i;
                }
                w.order[..dimension].sort_unstable_by(|&a, &b| {
                    compare(
                        schur.eigenvalues()[a],
                        schur.eigenvalues()[b],
                        self.options.selection,
                    )
                });
                let keep = wanted.max(dimension / 2).min(dimension - 2);
                w.mask[..dimension].fill(false);
                for &index in &w.order[..keep] {
                    w.mask[index] = true;
                }
                let report = schur.reorder_in_place(
                    &w.mask[..dimension],
                    &SchurReorderOptions::default().with_execution_options(
                        super::projected_execution_options(self.options.execution),
                    ),
                    schur_work,
                    work.kernels.scoped(&mut *work.solve),
                )?;
                let retained = report.selected_dimension();
                let beta = w.h.get_element(dimension, dimension - 1);
                krylov::combine_block(
                    crate::MatrixViewMut::from_slice(retained, n, &mut w.retained[..retained * n])?
                        .transpose(),
                    &w.basis[..dimension * n],
                    |j, target| schur.q().get_element(j, target),
                    &mut w.coefficients,
                    schur_work,
                    self.options.execution.execution_policy(),
                    work.kernels,
                )?;
                w.basis
                    .copy_within(dimension * n..(dimension + 1) * n, retained * n);
                w.basis[..retained * n].copy_from_slice(&w.retained[..retained * n]);
                w.h.as_mut_slice().fill(S::ZERO);
                for i in 0..retained {
                    for j in 0..retained {
                        w.h.as_mut_slice()[i * m + j] = schur.t().get_element(i, j);
                    }
                }
                for j in 0..retained {
                    w.h.as_mut_slice()[retained * m + j] =
                        beta * schur.q().get_element(dimension - 1, j);
                }
                Ok(retained)
            }
        }
    }
}
impl<R: RealScalar> PartialSpectrumSolver<R> {
    /// Checks the settings against the order `n` and returns the per-call value.
    fn call(&self, n: usize) -> Result<SpectrumCall<R>, SciOxideError> {
        super::validate_restart_capacity(n, self.count, self.basis)?;
        Ok(SpectrumCall { n, options: *self })
    }

    /// Returns the order of the square operator `A` of `op`.
    fn order<S: NumericScalar<Real = R>>(
        op: SpectrumOperator<'_, S>,
    ) -> Result<usize, SciOxideError> {
        let (SpectrumOperator::Direct(a) | SpectrumOperator::ShiftInvert { a, .. }) = op;
        crate::checked::square(a.rows(), a.cols())?;
        Ok(a.rows())
    }
    /// Returns the workspace storage needed to solve for the operator `op`.
    ///
    /// No operator is applied; only shapes and scratch lengths are read. The
    /// storage includes the dense projected eigensolver. The settings are
    /// checked first, so the execution methods report the same errors before
    /// any work.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `A` is not square or another operator of `op`
    ///   is not `n × n`.
    /// - `InvalidInput` (`OutOfRange`, argument `self.count`) if the requested
    ///   count exceeds `n`.
    /// - `InvalidInput` (`Inconsistent`, argument `self.basis`) if the basis
    ///   dimension is less than `count + 2` and less than `n`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of the operators.
    pub fn workspace_requirements<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
    ) -> Result<PartialSpectrumRequirements<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        self.call(Self::order(op)?)?.workspace_requirements(op)
    }

    /// Allocates a workspace for `op`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'o, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
    ) -> Result<PartialSpectrumWorkspace<S>, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        self.call(Self::order(op)?)?.workspace(op)
    }

    /// Computes the eigenpairs with a new workspace and result buffer.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`PartialSpectrumBuffer::new`] and
    /// [`Self::solve_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, PartialSpectrumSolver};
    /// let a = Matrix::<f64>::identity(2)?;
    /// let method = PartialSpectrumSolver::new(1);
    /// let (output, report) = method.solve(&a.view().try_finite()?, &[1., 0.][..])?;
    /// // The identity has the single eigenvalue 1.
    /// assert!((output.eigenvalues()[0].re - 1.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'o, 'i, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
        initial: impl Into<VectorView<'i, S>>,
    ) -> Result<(PartialSpectrumBuffer<R>, PartialSpectrumReport), SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        self.call(Self::order(op)?)?.solve(op, initial)
    }

    /// Computes eigenpairs of `op` from the starting vector `initial` and writes them to `result`.
    ///
    /// With `Hermitian` structure, the symmetry check of the projected matrix
    /// allows for the rounding error of its entries in addition to the residual
    /// tolerance.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `initial` does not have length `n`, or `result`
    ///   was allocated for another `n` or count.
    /// - `NonFinite` if `initial` or the shift is not finite, or if an operator
    ///   returns a non-finite value.
    /// - `InvalidInput` (`OutOfRange`) if `initial` is zero.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `InvalidInput` (`Inconsistent`) if the projected matrix shows that the
    ///   operator is not Hermitian (with `Hermitian` structure; this includes
    ///   shift-invert with a mass matrix that does not commute with `A`), or if
    ///   a scratch length changed after the workspace was reserved.
    /// - `NumericalInstability` (`Breakdown`) if a Ritz vector vanishes or, for
    ///   shift-invert, a transformed Ritz value is zero.
    /// - Errors of the dense eigensolver, Schur decomposition and reordering,
    ///   and errors returned by the operators.
    pub fn solve_into<'o, 'i, 's, 'c, S: NumericScalar<Real = R>>(
        &self,
        op: impl Into<SpectrumOperator<'o, S>>,
        initial: impl Into<VectorView<'i, S>>,
        result: &mut PartialSpectrumBuffer<R>,
        w: &mut PartialSpectrumWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<PartialSpectrumReport, SciOxideError>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let op = op.into();
        self.call(Self::order(op)?)?
            .solve_into(op, initial, result, w, context)
    }
}

fn compare<R: RealScalar>(
    a: Complex<R>,
    b: Complex<R>,
    selection: SpectrumSelection,
) -> core::cmp::Ordering {
    let (x, y) = match selection {
        SpectrumSelection::LargestMagnitude => (b.norm(), a.norm()),
        SpectrumSelection::SmallestMagnitude => (a.norm(), b.norm()),
        SpectrumSelection::SmallestReal => (a.re, b.re),
        SpectrumSelection::LargestReal => (b.re, a.re),
    };
    x.partial_cmp(&y)
        .unwrap_or(core::cmp::Ordering::Equal)
        .then_with(|| {
            a.im.partial_cmp(&b.im)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
}

impl<R: RealScalar> PartialSpectrumBuffer<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn storage_plan(
        &self,
        plan: &mut StoragePlan,
        n: usize,
        count: usize,
    ) -> Result<(), SciOxideError> {
        plan.matrix(&self.vectors, n, count)?;
        plan.buffer(&self.values, count)?;
        plan.buffer(&self.residuals, count)
    }
    fn configure(&mut self, n: usize, count: usize) -> Result<(), SciOxideError> {
        self.vectors.try_resize_flat(n, count)?;
        resize(&mut self.values, count, Complex::ZERO)?;
        resize(&mut self.residuals, count, R::ZERO)
    }
    const fn empty() -> Self {
        Self {
            values: Vec::new(),
            vectors: Matrix::empty(),
            residuals: Vec::new(),
            count: 0,
        }
    }
}

impl<R: RealScalar> StorageBytes for PartialSpectrumBuffer<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.values.storage_bytes()?,
            self.vectors.storage_bytes()?,
            self.residuals.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> Default for PartialSpectrumWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> PartialSpectrumWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            n: 0,
            m: 0,
            projection_roundoff: S::Real::ZERO,
            basis: Vec::new(),
            retained: Vec::new(),
            image: Vec::new(),
            coefficients: Vec::new(),
            real_input: Vec::new(),
            real_image: Vec::new(),
            complex_image: Vec::new(),
            provider: Vec::new(),
            h: Matrix::empty(),
            projected: Matrix::empty(),
            solver: ProjectedSolver::Hermitian {
                eigen: HermitianEigenDecomposition::empty(),
                work: LinalgWorkspace::new(),
            },
            other_solver: None,
            order: Vec::new(),
            mask: Vec::new(),
            candidate: PartialSpectrumBuffer::empty(),
            proposal: PartialSpectrumBuffer::empty(),
            shift_invert: None,
            stop: None,
            candidate_cycles: 0,
            proposal_cycles: 0,
            slot: SolveSlot::new(),
        }
    }
    /// Returns whether the workspace already holds the storage that `requirements` describes.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn check(
        &self,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        PlannedStorage::check(self, requirements)
    }
    /// Grows the workspace to the storage that `requirements` describes.
    ///
    /// Buffers that are already large enough keep their capacity, and the
    /// attempt record is kept. Storage for the Hermitian and the general
    /// projected solver, and for shift-invert, is kept when the workspace is
    /// reserved for another configuration, so switching back needs no new
    /// allocation. If growth fails, some buffers may already have grown; the
    /// workspace stays usable.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `AllocationFailure` if an allocation fails.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for PartialSpectrumWorkspace<S>
where
    scioxide_core::Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    type Requirements = PartialSpectrumRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let SpectrumLayout {
            n,
            basis: m,
            count,
            structure,
            provider,
            shift,
        } = requirements.layout;
        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.basis, product(&[n, sum([m, 1])?])?)?;
        plan.buffer(&self.retained, product(&[n, m])?)?;
        plan.buffer(&self.coefficients, product(&[m, m])?)?;
        plan.buffer(&self.image, n)?;
        plan.buffer(&self.real_input, n)?;
        plan.buffer(&self.real_image, n)?;
        plan.buffer(&self.complex_image, n)?;
        plan.buffer(&self.provider, provider)?;
        plan.buffer(&self.order, m)?;
        plan.buffer(&self.mask, m)?;
        plan.matrix(&self.h, sum([m, 1])?, m)?;
        plan.matrix(&self.projected, m, m)?;
        self.candidate.storage_plan(&mut plan, n, count)?;
        self.proposal.storage_plan(&mut plan, n, count)?;
        if self.solver.structure() == structure {
            self.solver.storage_plan(&mut plan, m, requirements)?;
        } else if let Some(other) = &self.other_solver {
            other.storage_plan(&mut plan, m, requirements)?;
        } else {
            ProjectedSolver::<S>::empty(structure).storage_plan(&mut plan, m, requirements)?;
        }
        if let Some(layout) = shift {
            self.shift_invert
                .as_ref()
                .unwrap_or(&ShiftInvertBuffers::empty())
                .storage_plan(&mut plan, n, layout)?;
        }
        Ok(plan)
    }
    fn configure(
        &mut self,
        requirements: &PartialSpectrumRequirements<S>,
    ) -> Result<(), SciOxideError> {
        let SpectrumLayout {
            n,
            basis: m,
            count,
            structure,
            provider,
            shift,
        } = requirements.layout;
        resize(&mut self.basis, product(&[n, sum([m, 1])?])?, S::ZERO)?;
        resize(&mut self.retained, product(&[n, m])?, S::ZERO)?;
        resize(&mut self.coefficients, product(&[m, m])?, S::ZERO)?;
        resize(&mut self.image, n, S::ZERO)?;
        resize(&mut self.real_input, n, S::ZERO)?;
        resize(&mut self.real_image, n, S::ZERO)?;
        resize(&mut self.complex_image, n, Complex::ZERO)?;
        resize(&mut self.provider, provider, S::ZERO)?;
        resize(&mut self.order, m, 0)?;
        resize(&mut self.mask, m, false)?;
        self.h.try_resize_flat(m + 1, m)?;
        self.projected.try_resize_flat(m, m)?;
        self.candidate.configure(n, count)?;
        self.proposal.configure(n, count)?;
        if self.solver.structure() != structure {
            let selected = self
                .other_solver
                .take()
                .unwrap_or_else(|| ProjectedSolver::empty(structure));
            self.other_solver = Some(core::mem::replace(&mut self.solver, selected));
        }
        self.solver.configure(m, requirements)?;
        if let Some(layout) = shift {
            self.shift_invert
                .get_or_insert_with(ShiftInvertBuffers::empty)
                .configure(n, layout)?;
        }
        self.n = n;
        self.m = m;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for PartialSpectrumWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn storage_bytes(&self) -> core::result::Result<usize, SciOxideError> {
        sum([
            self.basis.storage_bytes()?,
            self.retained.storage_bytes()?,
            self.image.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.real_input.storage_bytes()?,
            self.real_image.storage_bytes()?,
            self.complex_image.storage_bytes()?,
            self.provider.storage_bytes()?,
            self.h.storage_bytes()?,
            self.projected.storage_bytes()?,
            self.solver.storage_bytes()?,
            self.other_solver.storage_bytes()?,
            self.order.storage_bytes()?,
            self.mask.storage_bytes()?,
            self.candidate.storage_bytes()?,
            self.proposal.storage_bytes()?,
            self.shift_invert.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> StorageBytes for ProjectedSolver<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Hermitian { eigen, work } => sum([eigen.storage_bytes()?, work.storage_bytes()?]),
            Self::General {
                vectors,
                vector_work,
                phases,
                schur,
                schur_work,
            } => sum([
                vectors.storage_bytes()?,
                vector_work.storage_bytes()?,
                phases.storage_bytes()?,
                schur.storage_bytes()?,
                schur_work.storage_bytes()?,
            ]),
        }
    }
}

/// Termination status and work counts of a partial eigenvalue solve.
#[derive(Debug, Clone, Copy)]
pub struct PartialSpectrumReport {
    termination: SpectralTermination,
    applications: usize,
    cycles: usize,
}
impl PartialSpectrumReport {
    /// Returns the termination status.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }
    /// Returns the number of operator applications, including residual checks.
    pub const fn applications(&self) -> usize {
        self.applications
    }
    /// Returns the cycle, counted from 1, in which the returned pairs were
    /// computed, or zero if there are none.
    pub const fn cycles(&self) -> usize {
        self.cycles
    }
}
