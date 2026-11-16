//! Stabilizing solutions of continuous and discrete algebraic Riccati equations.
//!
//! Preparation forms the extended `(2n + m)`-order pencil, removes the control
//! unknowns with a QR factorization, computes the stable deflating subspace with
//! the QZ algorithm and reordering, and factors the denominator of its graph.
//! Solves recover `X` and the feedback gain and check the residual of the
//! original equation. Scalar equations with positive costs use a closed-form
//! root instead. A QZ check of the closed-loop spectrum is optional.
use super::blocked::{self, Products};
use crate::DirectSolveTermination;
use crate::eigen::qz_kernel::Pencil;
use crate::kernel::frames::{Framed, Grid, Input};
use crate::kernel::frames::{extended, scalar};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::{RealRegion, ScalarRegion};
use crate::workspace::{take_mut, take_square};
use crate::{
    DenseWorkspaceRequest, Estimation, GeneralizedEigenvalue, GeneralizedSchurDecomposition,
    GeneralizedSchurWorkspaceRequest, HouseholderDecomposition, HouseholderOrientation,
    HouseholderPivoting, KernelContext, LinalgExecutionContext, LinalgExecutionOptions,
    LinalgRequirements, LinalgWorkspace, LuDecomposition, Matrix, MatrixView, MatrixViewMut,
    SchurOptions, SchurReorderOptions, SvdValues,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, checked_product, try_filled};
use scioxide_core::{
    Budget, Dimensions, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar,
    ResidualTolerance, SciOxideError, SolveContext, Tolerance,
};
use scioxide_core::{Scaled, ScaledValue};

/// Continuous (CARE) or discrete (DARE) algebraic Riccati equation.
///
/// With the coefficients of a [`RiccatiProblem`] and the Hermitian unknown `X`,
/// the two equations in descriptor form are
///
/// ```text
/// CARE:  Aᴴ X E + Eᴴ X A − (Eᴴ X B + S) R⁻¹ (Bᴴ X E + Sᴴ) + Q = 0
/// DARE:  Aᴴ X A − Eᴴ X E − (Aᴴ X B + S) (R + Bᴴ X B)⁻¹ (Bᴴ X A + Sᴴ) + Q = 0
/// ```
///
/// The feedback gain `K`, with control law `u = −K x`, is
///
/// ```text
/// CARE:  K = R⁻¹ (Bᴴ X E + Sᴴ)
/// DARE:  K = (R + Bᴴ X B)⁻¹ (Bᴴ X A + Sᴴ)
/// ```
///
/// The stabilizing solution is the one for which every eigenvalue of the
/// closed-loop pencil `(A − B K, E)` lies in the open left half-plane (CARE) or
/// inside the open unit disk (DARE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiccatiKind {
    /// Continuous-time equation (CARE).
    Continuous,
    /// Discrete-time equation (DARE).
    Discrete,
}
/// Coefficients of an algebraic Riccati equation with `n` states and `m` controls.
///
/// The equations are stated on [`RiccatiKind`].
#[derive(Debug, Clone, Copy)]
pub struct RiccatiProblem<'a, S: NumericScalar> {
    /// State matrix `A`, `n × n`.
    pub a: MatrixView<'a, S>,
    /// Input matrix `B`, `n × m`.
    pub b: MatrixView<'a, S>,
    /// State cost `Q`, `n × n`, exactly Hermitian.
    pub q: MatrixView<'a, S>,
    /// Control cost `R`, `m × m`, exactly Hermitian, and nonsingular for CARE.
    pub r: MatrixView<'a, S>,
    /// Descriptor matrix `E`, `n × n` and nonsingular; `None` means `E = I`.
    pub e: Option<MatrixView<'a, S>>,
    /// Cross cost `S`, `n × m`; `None` means `S = 0`.
    pub s: Option<MatrixView<'a, S>>,
}

/// Diagonal scaling of the extended pencil before the QZ algorithm.
///
/// Scaling changes the state coordinates in which the pencil is reduced. The
/// solution is transformed back, so `X` and `K` are always in the original
/// coordinates. [`RiccatiPlan::state_scaling`] returns the state scale factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiccatiBalancing {
    /// No scaling.
    None,
    /// Power-of-two diagonal scaling that preserves the Hamiltonian or
    /// symplectic structure.
    ///
    /// The scaling first balances the row and column sums of the off-diagonal
    /// entries of `|H| + |J|`, where `H − λ J` is the extended pencil, with the
    /// Parlett–Reinsch iteration without permutations. It then sets the costate
    /// scales to the reciprocals of the state scales and keeps the control
    /// scales. Powers of two keep the zero pattern and add no rounding error
    /// unless an entry becomes subnormal.
    ///
    /// # References
    ///
    /// - B. N. Parlett and C. Reinsch, "Balancing a matrix for calculation of
    ///   eigenvalues and eigenvectors", *Numerische Mathematik* 13, 1969.
    /// - P. Benner, "Symplectic balancing of Hamiltonian matrices", *SIAM J.
    ///   Sci. Comput.* 22(5), 2001, DOI: 10.1137/S1064827500367993.
    Scale,
}

/// Options of a [`RiccatiPlan`].
///
/// A solve is accepted when the residual of the original equation satisfies
///
/// ```text
/// ‖Res‖_F ≤ atol + rtol · (‖T₁‖_F + ‖T₂‖_F + ‖T₃‖_F + ‖T₄‖_F),
/// ```
///
/// where `Res = T₁ + T₂ + T₃ + T₄` and the terms are, with the computed `X` and `K`,
/// `T₁ = Aᴴ X E`, `T₂ = Eᴴ X A`, `T₃ = −(Eᴴ X B + S) K`, `T₄ = Q` for CARE and
/// `T₁ = Aᴴ X A`, `T₂ = −Eᴴ X E`, `T₃ = −(Aᴴ X B + S) K`, `T₄ = Q` for DARE.
///
/// An eigenvalue `λ = α/β` of a pencil has the stability margin
/// `−Re(α) / max(|α|, |β|)` for CARE and `(|β| − |α|) / max(|α|, |β|)` for
/// DARE. An infinite eigenvalue has margin −1. Preparation fails if an
/// eigenvalue of the extended pencil has `|margin| ≤ atol + rtol`, with `atol`
/// and `rtol` from the stability tolerance.
///
/// Defaults: residual tolerance `atol = rtol = √ε`, stability tolerance
/// `atol = rtol = ε` ([`Tolerance::from_machine_epsilon`]),
/// [`RiccatiBalancing::Scale`], and no closed-loop verification. The QZ
/// iteration and the reordering may each use 30 000 iterations, or the
/// iteration budget of the execution options when one is set.
///
/// ```
/// use scioxide_linalg::{LinalgExecutionOptions, RiccatiBalancing, RiccatiOptions};
/// let execution = LinalgExecutionOptions::new().with_byte_limit(1 << 20);
/// let options = RiccatiOptions::<f64>::new()
///     .with_execution_options(execution)
///     .with_balancing(RiccatiBalancing::None);
/// assert_eq!(options.execution_options(), execution);
/// assert_eq!(options.balancing(), RiccatiBalancing::None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RiccatiOptions<R: RealScalar> {
    execution: LinalgExecutionOptions,
    tolerance: ResidualTolerance<R>,
    stability_tolerance: Tolerance<R>,
    balancing: RiccatiBalancing,
    closed_loop_verification: Estimation,
}
impl<R: RealScalar> Default for RiccatiOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> RiccatiOptions<R> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            tolerance: ResidualTolerance::from_tolerance(Tolerance::from_sqrt_epsilon()),
            stability_tolerance: Tolerance::from_machine_epsilon(),
            balancing: RiccatiBalancing::Scale,
            closed_loop_verification: Estimation::Skip,
        }
    }
    /// Sets the residual tolerance.
    ///
    /// The same tolerance bounds the Hermitian defect of the recovered `X`:
    /// a solve fails if `maxᵢⱼ |xᵢⱼ − conj(xⱼᵢ)| > atol + rtol · maxᵢⱼ |xᵢⱼ|`.
    pub const fn with_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Sets the minimum stability margin of the eigenvalues, defined on
    /// [`RiccatiOptions`], measured from the imaginary axis (CARE) or the unit
    /// circle (DARE).
    pub const fn with_stability_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.stability_tolerance = tolerance;
        self
    }
    /// Selects whether each solve checks the eigenvalues of the closed-loop pencil.
    ///
    /// With [`Estimation::Estimate`], a solve computes the eigenvalues of
    /// `(A − B K, E)` with the QZ algorithm, which costs O(n³) operations. It
    /// reports their smallest margin in [`RiccatiReport::stability_margin`] and
    /// returns [`DirectSolveTermination::StabilityRejected`] if a margin is at or
    /// below the stability tolerance. The default, [`Estimation::Skip`], relies
    /// on the stable subspace chosen during preparation and checks only the
    /// residual and the Hermitian defect.
    ///
    /// ```
    /// use scioxide_linalg::{Estimation, RiccatiOptions};
    /// let options =
    ///     RiccatiOptions::<f64>::new().with_closed_loop_verification(Estimation::Estimate);
    /// assert_eq!(options.closed_loop_verification(), Estimation::Estimate);
    /// ```
    pub const fn with_closed_loop_verification(mut self, verification: Estimation) -> Self {
        self.closed_loop_verification = verification;
        self
    }
    /// Returns whether solves check the closed-loop eigenvalues.
    pub const fn closed_loop_verification(self) -> Estimation {
        self.closed_loop_verification
    }
    /// Sets the scaling of the extended pencil.
    pub const fn with_balancing(mut self, balancing: RiccatiBalancing) -> Self {
        self.balancing = balancing;
        self
    }
    /// Sets the execution options used by preparation and solves.
    pub const fn with_execution_options(mut self, execution: LinalgExecutionOptions) -> Self {
        self.execution = execution;
        self
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> LinalgExecutionOptions {
        self.execution
    }
    /// Returns the residual tolerance.
    pub const fn tolerance(self) -> ResidualTolerance<R> {
        self.tolerance
    }
    /// Returns the stability tolerance.
    pub const fn stability_tolerance(self) -> Tolerance<R> {
        self.stability_tolerance
    }
    /// Returns the pencil scaling.
    pub const fn balancing(self) -> RiccatiBalancing {
        self.balancing
    }
    /// Iteration limit of the QZ iteration and of the reordering.
    fn stage_iteration_budget(self) -> Budget {
        self.execution
            .iteration_budget()
            .unwrap_or(Budget::ONE.saturating_add(29_999))
    }
}
/// Diagnostics of one Riccati solve.
///
/// The residual is computed from the original coefficients, not from the
/// scaled pencil.
#[derive(Debug, Clone, Copy)]
pub struct RiccatiReport<R: RealScalar> {
    /// Acceptance status.
    pub(crate) termination: DirectSolveTermination,
    /// Original-equation Frobenius residual.
    pub(crate) residual: R,
    /// Residual divided by the sum of equation-term Frobenius norms.
    pub(crate) backward_error: R,
    /// Maximum entrywise Hermitian defect before projection.
    pub(crate) symmetry_defect: R,
    /// Reciprocal condition estimate of the descriptor-adjusted graph denominator;
    /// one for analytical scalar recovery, which has no graph solve.
    pub(crate) subspace_reciprocal_condition: R,
    /// Minimum normalized homogeneous closed-loop stability margin; absent when
    /// independent verification is disabled or the state is empty.
    pub(crate) stability_margin: Option<R>,
    /// Solve stages and nested iterations, excluding preparation.
    pub(crate) work: usize,
}

impl<R: RealScalar> RiccatiReport<R> {
    /// Returns the acceptance status.
    ///
    /// It is [`DirectSolveTermination::Accepted`],
    /// [`DirectSolveTermination::ResidualRejected`] if the residual test failed,
    /// or [`DirectSolveTermination::StabilityRejected`] if the optional
    /// closed-loop check failed.
    pub const fn termination(&self) -> DirectSolveTermination {
        self.termination
    }

    /// Returns the Frobenius norm `‖Res‖_F` of the original-equation residual.
    pub const fn residual(&self) -> R {
        self.residual
    }

    /// Returns `‖Res‖_F / (‖T₁‖_F + ‖T₂‖_F + ‖T₃‖_F + ‖T₄‖_F)`, or zero when all
    /// terms vanish.
    ///
    /// The terms are defined on [`RiccatiOptions`].
    pub const fn backward_error(&self) -> R {
        self.backward_error
    }

    /// Returns `maxᵢⱼ |xᵢⱼ − conj(xⱼᵢ)|` of the recovered `X`, before `X` is
    /// replaced by its Hermitian part `(X + Xᴴ)/2`.
    ///
    /// It is zero for the closed-form scalar solution.
    pub const fn symmetry_defect(&self) -> R {
        self.symmetry_defect
    }

    /// Returns a measure of how well the stable subspace determines `X`, between
    /// 0 and 1.
    ///
    /// The stable deflating subspace is spanned by the columns of `[U₁; U₂]`
    /// with orthonormal columns, and `X = U₂ (E U₁)⁻¹` in the scaled
    /// coordinates. The value is the smaller of a one-norm reciprocal condition
    /// estimate of `E U₁` and the smallest singular value of `U₁`. Preparation
    /// fails if it is at most `ε`. It is 1 for the closed-form scalar solution.
    pub const fn subspace_reciprocal_condition(&self) -> R {
        self.subspace_reciprocal_condition
    }

    /// Returns the smallest stability margin of the closed-loop eigenvalues.
    ///
    /// The margin is defined on [`RiccatiOptions`]. It is `None` when
    /// closed-loop verification is disabled or `n = 0`.
    pub const fn stability_margin(&self) -> Option<R> {
        self.stability_margin
    }

    /// Returns the number of solve stages and nested iterations, including the
    /// optional closed-loop QZ and excluding preparation.
    pub const fn work(&self) -> usize {
        self.work
    }
}

/// Stabilizing solution `X`, feedback gain `K` and diagnostics of a Riccati solve.
///
/// `X` and `K` are available only when the solve was accepted
/// (see [direct solves](crate#direct-solves)).
#[derive(Debug)]
pub struct RiccatiResult<S: NumericScalar> {
    /// Hermitian solution in original state coordinates.
    pub(crate) solution: Matrix<S>,
    /// Control-by-state feedback gain.
    pub(crate) gain: Matrix<S>,
    /// Diagnostics of the solve.
    pub(crate) report: RiccatiReport<S::Real>,
}

impl<S: NumericScalar> RiccatiResult<S> {
    /// Returns the accepted pair `(X, K)`, or `None` if it was rejected, and the report.
    #[expect(
        clippy::type_complexity,
        reason = "solution and gain are accepted together"
    )]
    pub fn into_parts(self) -> (Option<(Matrix<S>, Matrix<S>)>, RiccatiReport<S::Real>) {
        (
            self.report
                .termination
                .is_success()
                .then_some((self.solution, self.gain)),
            self.report,
        )
    }

    /// Returns the accepted `n × n` Hermitian solution `X`, or `None` if it was
    /// rejected.
    pub fn solution(&self) -> Option<&Matrix<S>> {
        self.report
            .termination
            .is_success()
            .then_some(&self.solution)
    }

    /// Returns the accepted `m × n` feedback gain `K`, or `None` if it was rejected.
    pub fn gain(&self) -> Option<&Matrix<S>> {
        self.report.termination.is_success().then_some(&self.gain)
    }

    /// Returns the accepted pair `(X, K)`, or the error of
    /// [`DirectSolveTermination::require_accepted`] for a rejected solve.
    ///
    /// # Errors
    ///
    /// `NumericalInstability` with `IllConditioned` if the closed loop is not
    /// stable within the tolerance, or with `LossOfSignificance` if the
    /// residual test rejected the solution.
    pub fn into_accepted(self) -> Result<(Matrix<S>, Matrix<S>), SciOxideError> {
        self.report.termination.require_accepted()?;
        Ok((self.solution, self.gain))
    }

    /// Returns the diagnostics of the solve.
    pub const fn report(&self) -> RiccatiReport<S::Real> {
        self.report
    }
}

/// Algebraic Riccati equation with fixed coefficients, prepared for solving.
///
/// The plan computes the stabilizing solution `X` and the feedback gain `K` of
/// the CARE or DARE stated on [`RiccatiKind`]. The plan does not check
/// existence conditions such as stabilizability and detectability. It fails
/// instead when the computed spectrum does not split into `n` stable and `n`
/// unstable eigenvalues separated by the stability tolerance. `E` must be
/// nonsingular.
///
/// Preparation uses the generalized Schur method on the extended pencil
/// `H − λ J` of order `2n + m`, written here by block rows:
///
/// ```text
/// CARE:  H = [ A  0  B ; −Q  −Aᴴ  −S ; Sᴴ  Bᴴ  R ]
///        J = [ E  0  0 ;  0   Eᴴ   0 ;  0   0  0 ]
///
/// DARE:  H = [ A  0  B ; −Q   Eᴴ  −S ; Sᴴ   0  R ]
///        J = [ E  0  0 ;  0   Aᴴ   0 ;  0 −Bᴴ  0 ]
/// ```
///
/// The control unknowns are kept in the pencil, so `R⁻¹` and
/// `(R + Bᴴ X B)⁻¹` are never formed. Preparation optionally scales the pencil
/// ([`RiccatiBalancing`]), removes the last `m` columns with a column-pivoted
/// QR factorization of `[B; −S; R]`, reduces the remaining `2n × 2n` pencil to
/// generalized Schur form with the QZ algorithm, and reorders the `n` stable
/// eigenvalues (real part `< 0` for CARE, modulus `< 1` for DARE) to the top.
/// With the leading `n` columns `[U₁; U₂]` of the right Schur vectors,
/// `X = U₂ (E U₁)⁻¹` in the scaled coordinates, and the plan stores an LU
/// factorization of `(E U₁)ᴴ`.
/// Preparation costs O((n + m)³) operations.
///
/// Each solve recovers `X`, checks that it is Hermitian to within the residual
/// tolerance and replaces it by `(X + Xᴴ)/2`, computes `K`, and checks the
/// residual of the original equation as described on [`RiccatiOptions`].
///
/// A scalar equation (`n = m = 1`) with `E = 1`, `S = 0`, `Q > 0`, `R > 0` and
/// `B ≠ 0` is solved instead from the positive root of its quadratic, with the
/// same stability test.
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # References
///
/// - A. J. Laub, "A Schur method for solving algebraic Riccati equations",
///   *IEEE Trans. Automat. Control* 24(6), 1979.
/// - P. Van Dooren, "A generalized eigenvalue approach for solving Riccati
///   equations", *SIAM J. Sci. Stat. Comput.* 2(2), 1981.
/// - W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and
///   software for algebraic Riccati equations", *Proc. IEEE* 72(12), 1984.
#[derive(Debug, Clone)]
pub struct RiccatiPlan<S: NumericScalar> {
    a: Matrix<S>,
    b: Matrix<S>,
    q: Matrix<S>,
    r: Matrix<S>,
    e: Matrix<S>,
    s: Matrix<S>,
    recovery: Recovery<S>,
    scales: Vec<S::Real>,
    condition: S::Real,
    kind: RiccatiKind,
    options: RiccatiOptions<S::Real>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    target_pointer_width = "64",
    expect(
        clippy::large_enum_variant,
        reason = "inline retained LU storage keeps recovery allocation-free"
    )
)]
enum Recovery<S: NumericScalar> {
    Scalar(S::Real),
    Graph {
        denominator: LuDecomposition<S>,
        rhs: Matrix<S>,
    },
}

/// Owned copies of validated coefficients used during preparation.
struct RiccatiCoefficients<S: NumericScalar> {
    a: Matrix<S>,
    b: Matrix<S>,
    q: Matrix<S>,
    r: Matrix<S>,
    e: Matrix<S>,
    s: Matrix<S>,
}

/// Graph recovery data isolated from the ordered stable deflating subspace.
struct StableGraph<S: NumericScalar> {
    /// `(E' U1)^H` with `E' = D^-1 E D`.
    denominator: Matrix<S>,
    /// Leading `U1` block of the right deflating subspace.
    leading: Matrix<S>,
    /// `U2^H`.
    rhs: Matrix<S>,
}

impl<S: NumericScalar> RiccatiPlan<S> {
    /// Prepares the plan for `problem`.
    ///
    /// This is the owned form of [`Self::build`] (see [execution](crate#execution)).
    ///
    /// # Errors
    /// - The errors of [`Self::build`].
    /// - `CapabilityUnavailable` if the options set a time limit.
    pub fn new(
        problem: RiccatiProblem<'_, S>,
        kind: RiccatiKind,
        options: RiccatiOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            problem,
            kind,
            options,
            ExecutionPolicy::Sequential,
            ExecutionContext::sequential_native().into(),
        )
    }

    /// Prepares the plan for `problem` under `context`.
    ///
    /// The QZ iterations, the reordering and the other preparation stages count
    /// as nested work of `context`. The plan does not keep `context`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - `DimensionMismatch` if `A`, `Q` or `E` is not `n × n`, `B` or `S` is
    ///   not `n × m`, or `R` is not `m × m`, where `n` is the number of rows of
    ///   `A` and `m` the number of columns of `B`.
    /// - `NonFinite` if a coefficient has a NaN or infinite entry.
    /// - `InvalidInput` (argument `q` or `r`) if `Q` or `R` is not exactly
    ///   Hermitian.
    /// - `Singular` if the LU factorization of `E`, or of `R` for CARE, meets an
    ///   exactly zero pivot.
    /// - `NumericalInstability` (`IllConditioned`) if an eigenvalue of the pencil
    ///   lies within the stability tolerance of the imaginary axis (CARE) or the
    ///   unit circle (DARE), if the number of stable eigenvalues is not `n`, if
    ///   the pencil has an eigenvalue `0/0`, or if
    ///   [`RiccatiReport::subspace_reciprocal_condition`] would be at most `ε`.
    /// - `ConvergenceFailure` if the QZ iteration, the reordering or an SVD does
    ///   not converge within its iteration limit.
    /// - `ResourceLimit` or `AllocationFailure` for the preparation's storage.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn build<'scope, 'capability>(
        problem: RiccatiProblem<'_, S>,
        kind: RiccatiKind,
        options: RiccatiOptions<S::Real>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            problem,
            kind,
            options,
            options.execution_options().execution_policy(),
            context.into(),
        )
    }

    fn prepare(
        problem: RiccatiProblem<'_, S>,
        kind: RiccatiKind,
        options: RiccatiOptions<S::Real>,
        policy: ExecutionPolicy,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let scope = context.with_options(&options.execution)?;
        scope.kernel().execution().check()?;
        let (n, m) = validate_problem_shape(&problem)?;
        let coefficients = RiccatiCoefficients {
            a: problem.a.try_to_matrix()?,
            b: problem.b.try_to_matrix()?,
            q: problem.q.try_to_matrix()?,
            r: problem.r.try_to_matrix()?,
            e: if let Some(v) = problem.e {
                crate::checked::validate_matrix(v, n, n)?;
                v.try_to_matrix()?
            } else {
                Matrix::identity(n)?
            },
            s: if let Some(v) = problem.s {
                crate::checked::validate_matrix(v, n, m)?;
                v.try_to_matrix()?
            } else {
                Matrix::zeros(n, m)?
            },
        };
        let decomposition = options.execution.with_execution_policy(policy);
        let mut workspace = LinalgWorkspace::new();
        workspace.run_scope(
            scope,
            identity(n, m),
            options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                stage(solve)?;
                let mut graph = LuDecomposition::empty();
                if problem.e.is_some() {
                    LuDecomposition::factor_into(
                        &coefficients.e,
                        &decomposition,
                        &mut graph,
                        workspace,
                        kernels.scoped(&mut *solve),
                    )?;
                }
                if kind == RiccatiKind::Continuous {
                    LuDecomposition::factor_into(
                        &coefficients.r,
                        &decomposition,
                        &mut graph,
                        workspace,
                        kernels.scoped(&mut *solve),
                    )?;
                }
                if let Some(value) = scalar_solution(&coefficients, kind, &options)? {
                    solve.execution().check()?;
                    let RiccatiCoefficients { a, b, q, r, e, s } = coefficients;
                    return Ok(Self {
                        a,
                        b,
                        q,
                        r,
                        e,
                        s,
                        recovery: Recovery::Scalar(value),
                        scales: try_filled(n, S::Real::ONE, scioxide_core::quantity!("scales"))?,
                        condition: S::Real::ONE,
                        kind,
                        options,
                    });
                }
                // Extended (2n+m) pencil (Van Dooren): the
                // control is kept as explicit unknowns instead of forming `R^{-1}` or
                // `(R + B^H X B)^{-1}`, which avoids inverting an ill-conditioned R.
                // CARE: `[A 0 B; -Q -A^H -S; S^H B^H R] - lambda [E 0 0; 0 E^H 0; 0 0 0]`;
                // DARE uses the symplectic arrangement with A and E exchanged in the
                // costate block and `-B^H` moved into the J side.
                let nn = checked_mul(n, 2, scioxide_core::quantity!("pencil_order"))?;
                let total = checked_add(nn, m, scioxide_core::quantity!("pencil_order"))?;
                let (mut h, mut j) = extended_pencil(&coefficients, kind, (nn, total))?;
                let scales = balance(&mut h, &mut j, n, options.balancing, solve)?;
                let (hp, jp) = deflate_control(
                    (&h, &j),
                    m,
                    &decomposition,
                    (workspace, kernels, &mut *solve),
                )?;
                let subspace = stable_graph(
                    (&hp, &jp),
                    (problem.e.map(|_| &coefficients.e), &scales),
                    kind,
                    (&options, policy),
                    workspace,
                    kernels,
                    solve,
                )?;
                stage(solve)?;
                // The stable deflating subspace `[U1; U2]` gives `X' E' U1 = U2`, i.e.
                // `X' = U2 (E' U1)^{-1}`; the LU of `(E' U1)^H` is stored so X is later
                // recovered by a solve. The smaller of its reciprocal condition and
                // `sigma_min(U1)` measures how well the graph representation exists.
                LuDecomposition::factor_into(
                    &subspace.denominator,
                    &decomposition,
                    &mut graph,
                    workspace,
                    kernels.scoped(&mut *solve),
                )?;
                let mut graph_values = SvdValues::empty();
                SvdValues::compute_into(
                    &subspace.leading,
                    &decomposition,
                    &mut graph_values,
                    workspace,
                    kernels.scoped(solve),
                )?;
                let condition = if n == 0 {
                    S::Real::ONE
                } else {
                    graph
                        .condition_estimate_scoped(
                            &options.execution,
                            workspace,
                            kernels.scoped(&mut *solve),
                        )?
                        .reciprocal()
                        .value()
                        .min(graph_values.singular_values()[n - 1])
                };
                if condition <= <S::Real as RealScalar>::EPSILON {
                    return Err(instability(
                        "stable graph denominator is numerically singular",
                    ));
                }
                solve.execution().check()?;
                let RiccatiCoefficients { a, b, q, r, e, s } = coefficients;
                Ok(Self {
                    a,
                    b,
                    q,
                    r,
                    e,
                    s,
                    recovery: Recovery::Graph {
                        denominator: graph,
                        rhs: subspace.rhs,
                    },
                    scales,
                    condition,
                    kind,
                    options,
                })
            },
        )
    }

    /// Returns the diagonal `d` of the state scaling `x = D z`, where `z` is the
    /// state in the scaled pencil.
    ///
    /// All entries are 1 with [`RiccatiBalancing::None`] and for the closed-form
    /// scalar solution. `X` and `K` are always returned in the original coordinates.
    pub fn state_scaling(&self) -> &[S::Real] {
        &self.scales
    }

    /// Returns the workspace size needed by [`Self::solve_into`].
    ///
    /// The size does not include packing storage for a BLAS provider. When
    /// the context of [`Self::solve_into`] has a provider, the workspace grows
    /// at the call if needed.
    ///
    /// # Errors
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `ResourceLimit` if it exceeds the configured byte limit.
    pub fn workspace_requirements(&self, (): ()) -> Result<LinalgRequirements, SciOxideError> {
        requirements::<S>(
            (self.a.rows(), self.b.cols()),
            &self.options.execution,
            self.options.execution_options().execution_policy(),
            None,
            self.options.closed_loop_verification,
        )
    }

    /// Allocates a workspace for [`Self::solve_into`].
    ///
    /// # Errors
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the allocation fails.
    pub fn workspace(&self, (): ()) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        Ok(workspace)
    }

    /// Computes the stabilizing solution `X` and the feedback gain `K`.
    ///
    /// This is the owned form of [`Self::solve_into`]
    /// (see [execution](crate#execution)).
    ///
    /// # Errors
    /// - The errors of [`Self::solve_into`].
    /// - `CapabilityUnavailable` if the options set a time limit.
    pub fn solve(&self) -> Result<RiccatiResult<S>, SciOxideError> {
        let (n, m) = (self.a.rows(), self.b.cols());
        let mut solution = Matrix::zeros(n, n)?;
        let mut gain = Matrix::zeros(m, n)?;
        let report = self.run(
            solution.view_mut(),
            gain.view_mut(),
            ExecutionPolicy::Sequential,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native().into(),
        )?;
        Ok(RiccatiResult {
            solution,
            gain,
            report,
        })
    }

    /// Computes `X` and `K` and writes them to the `n × n` view `solution`
    /// and the `m × n` view `gain`.
    ///
    /// If the residual test or the optional closed-loop check fails, the call
    /// returns `Ok` with [`DirectSolveTermination::ResidualRejected`] or
    /// [`DirectSolveTermination::StabilityRejected`] and writes neither output
    /// (see [direct solves](crate#direct-solves)). Solve stages and QZ
    /// iterations count as nested work of `context`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - `DimensionMismatch` if `solution` is not `n × n` or `gain` is not `m × n`.
    /// - `NumericalInstability` (`IllConditioned`) if the recovered `X` is not
    ///   Hermitian within the residual tolerance (see
    ///   [`RiccatiOptions::with_tolerance`]).
    /// - `NumericalInstability` (`Overflow`) if `X`, `K` or a diagnostic
    ///   overflows the scalar type.
    /// - `Singular` if the LU factorization of `R` (CARE) or `R + Bᴴ X B` (DARE)
    ///   meets an exactly zero pivot.
    /// - `ConvergenceFailure` if the closed-loop QZ iteration does not converge
    ///   within its iteration limit, or the iteration budget is exhausted.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn solve_into<'scope, 'capability>(
        &self,
        solution: MatrixViewMut<'_, S>,
        gain: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<RiccatiReport<S::Real>, SciOxideError> {
        self.run(
            solution,
            gain,
            self.options.execution_options().execution_policy(),
            workspace,
            context.into(),
        )
    }

    fn run(
        &self,
        mut solution: MatrixViewMut<'_, S>,
        mut gain: MatrixViewMut<'_, S>,
        policy: ExecutionPolicy,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<RiccatiReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.with_options(&self.options.execution)?;
        let kernels = scope.kernel();
        kernels.execution().check()?;
        let (n, m) = (self.a.rows(), self.b.cols());
        for (rows, cols, actual) in [
            (n, n, (solution.rows(), solution.cols())),
            (m, n, (gain.rows(), gain.cols())),
        ] {
            if actual != (rows, cols) {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::two(rows, cols),
                    Dimensions::two(actual.0, actual.1),
                ));
            }
        }
        let requirements = requirements::<S>(
            (n, m),
            &self.options.execution,
            policy,
            Some(kernels),
            self.options.closed_loop_verification,
        )?;
        workspace.try_reserve(&requirements)?;
        let factor_lengths = crate::decompose::factor_kernel_lengths(m)?;
        workspace.run_scope(
            scope,
            identity(n, m),
            self.options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                workspace.with_direct(|mut scalars, mut reals, indices, gemm| {
                    let mut w = Staging {
                        x: take_square(&mut scalars, n)?,
                        xe: take_square(&mut scalars, n)?,
                        xa: take_square(&mut scalars, n)?,
                        xb: carve_matrix(&mut scalars, n, m)?,
                        rhs: carve_matrix(&mut scalars, m, n)?,
                        denominator: take_square(&mut scalars, m)?,
                        gain: carve_matrix(&mut scalars, m, n)?,
                        factor: take_mut(&mut scalars, factor_lengths.scalars)?,
                        reduction: take_mut(
                            &mut scalars,
                            if self.options.closed_loop_verification == Estimation::Estimate {
                                crate::eigen::qz_kernel::scratch_len::<S>(n, policy)?
                            } else {
                                0
                            },
                        )?,
                        indices,
                        values: take_mut(
                            &mut reals,
                            if self.options.closed_loop_verification == Estimation::Estimate {
                                3 * n
                            } else {
                                0
                            },
                        )?,
                        packing: take_mut(&mut scalars, blocked::PACKING)?,
                        residual_values: take_mut(&mut scalars, 3 * blocked::ELEMENTS)?,
                        gemm,
                        policy,
                        kernels,
                    };
                    let report = self.verified_candidate(&mut w, solve)?;
                    solve.execution().check()?;
                    if report.termination.is_success() {
                        for i in 0..n {
                            for j in 0..n {
                                solution.set_element(i, j, w.x.get_element(i, j));
                            }
                        }
                        for i in 0..m {
                            for j in 0..n {
                                gain.set_element(i, j, w.gain.get_element(i, j));
                            }
                        }
                    }
                    Ok(report)
                })
            },
        )
    }

    fn verified_candidate(
        &self,
        w: &mut Staging<'_, '_, '_, S>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<RiccatiReport<S::Real>, SciOxideError> {
        let n = self.a.rows();
        let defect = self.recover_graph(w, solve)?;
        self.recover_feedback(w, solve)?;
        stage(solve)?;
        let (residual, scale_norm) = self.residual(w, solve)?;
        let threshold = Scaled::from_value(self.options.tolerance.absolute())?
            .add(Scaled::from_value(self.options.tolerance.relative())?.mul(scale_norm)?)?;
        let mut termination = if residual.abs_le(threshold) {
            DirectSolveTermination::Accepted
        } else {
            DirectSolveTermination::ResidualRejected
        };
        // Preparation selects the stabilizing subspace. A fresh closed-loop QZ
        // is an optional, independent check of the recovered candidate.
        let mut stability_margin = None;
        if self.options.closed_loop_verification == Estimation::Estimate {
            stage(solve)?;
            multiply(
                self.b.view(),
                w.gain.try_view()?,
                w.xe.try_view_mut()?,
                &self.options.execution.with_execution_policy(w.policy),
                w.gemm,
                w.kernels,
            )?;
            for (v, &a) in w.xe.as_mut_slice().iter_mut().zip(self.a.as_slice()) {
                *v = crate::checked::sub(a, *v)?;
            }
            w.xa.as_mut_slice().copy_from_slice(self.e.as_slice());
            let qz_options = SchurOptions::new().with_execution_options(
                self.options
                    .execution
                    .with_execution_policy(w.policy)
                    .with_iteration_budget(self.options.stage_iteration_budget()),
            );
            crate::eigen::qz_values(
                w.xe.as_mut_slice(),
                w.xa.as_mut_slice(),
                n,
                w.reduction,
                w.values,
                &qz_options,
                solve,
            )?;
            let threshold = self.options.stability_tolerance.threshold(S::Real::ONE)?;
            let mut stable = true;
            for parts in w.values.chunks_exact(3) {
                let d = margin(GeneralizedEigenvalue::from_parts(parts), self.kind)?;
                stability_margin = Some(stability_margin.map_or(d, |x: S::Real| x.min(d)));
                if d <= threshold {
                    stable = false;
                }
            }
            if !stable {
                termination = DirectSolveTermination::StabilityRejected;
            }
        }
        Ok(RiccatiReport {
            termination,
            residual: residual.round()?,
            backward_error: if scale_norm.is_zero() {
                S::Real::ZERO
            } else {
                residual.div(scale_norm)?.round()?
            },
            symmetry_defect: defect,
            subspace_reciprocal_condition: self.condition,
            stability_margin,
            work: solve.counts().total()?.iterations(),
        })
    }

    fn recover_graph(
        &self,
        w: &mut Staging<'_, '_, '_, S>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<S::Real, SciOxideError> {
        let n = self.a.rows();
        stage(solve)?;
        let (denominator, rhs) = match &self.recovery {
            Recovery::Scalar(value) => {
                w.x.as_mut_slice()[0] = S::from_real(*value);
                return Ok(S::Real::ZERO);
            }
            Recovery::Graph { denominator, rhs } => (denominator, rhs),
        };
        denominator.solve_matrix_candidate(rhs.view(), w.x.as_mut_slice(), w.gemm, w.kernels)?;
        // X = D^-1 X'^H D^-1 restores the original state coordinates.
        for i in 0..n {
            for j in i..n {
                let u = scale(
                    scale(w.x.get_element(j, i).conj(), S::Real::ONE, self.scales[i])?,
                    S::Real::ONE,
                    self.scales[j],
                )?;
                let v = scale(
                    scale(w.x.get_element(i, j).conj(), S::Real::ONE, self.scales[j])?,
                    S::Real::ONE,
                    self.scales[i],
                )?;
                w.x.as_mut_slice()[i * n + j] = u;
                w.x.as_mut_slice()[j * n + i] = v;
            }
        }
        // The exact stabilizing solution is Hermitian; the computed one is only
        // approximately so. A defect beyond tolerance indicates a wrong or
        // ill-conditioned subspace and is rejected; otherwise X is projected
        // onto the Hermitian matrices by averaging with its adjoint.
        let mut defect = S::Real::ZERO;
        let mut magnitude = S::Real::ZERO;
        for i in 0..n {
            for j in i..n {
                defect = defect.max((w.x.get_element(i, j) - w.x.get_element(j, i).conj()).abs());
                magnitude = magnitude.max(w.x.get_element(i, j).abs());
            }
        }
        if defect > self.options.tolerance.threshold(magnitude)? {
            return Err(instability("stable graph is not numerically Hermitian"));
        }
        for i in 0..n {
            for j in i..n {
                let average = scalar::<S>(
                    extended(w.x.get_element(i, j))?
                        .add(extended(w.x.get_element(j, i).conj())?)?
                        .mul(ScaledValue::from_value(scioxide_core::Complex::new(
                            S::Real::HALF,
                            <_ as scioxide_core::AlgebraicScalar>::ZERO,
                        ))?)?,
                )?;
                w.x.as_mut_slice()[i * n + j] = average;
                w.x.as_mut_slice()[j * n + i] = average.conj();
            }
        }
        Ok(defect)
    }

    fn recover_feedback(
        &self,
        w: &mut Staging<'_, '_, '_, S>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.a.rows();
        let m = self.b.cols();
        for (right, target) in [(self.e.view(), &mut w.xe), (self.a.view(), &mut w.xa)] {
            stage(solve)?;
            multiply(
                w.x.try_view()?,
                right,
                target.try_view_mut()?,
                &self.options.execution.with_execution_policy(w.policy),
                w.gemm,
                w.kernels,
            )?;
        }
        stage(solve)?;
        multiply(
            w.x.try_view()?,
            self.b.view(),
            w.xb.try_view_mut()?,
            &self.options.execution.with_execution_policy(w.policy),
            w.gemm,
            w.kernels,
        )?;
        let target = if self.kind == RiccatiKind::Continuous {
            w.xe.try_view()?
        } else {
            w.xa.try_view()?
        };
        stage(solve)?;
        multiply(
            self.b.view().adjoint(),
            target,
            w.rhs.try_view_mut()?,
            &self.options.execution.with_execution_policy(w.policy),
            w.gemm,
            w.kernels,
        )?;
        for i in 0..m {
            for j in 0..n {
                w.rhs.as_mut_slice()[i * n + j] =
                    crate::checked::add(w.rhs.get_element(i, j), self.s.get_element(j, i).conj())?;
            }
        }
        if self.kind == RiccatiKind::Discrete {
            multiply(
                self.b.view().adjoint(),
                w.xb.try_view()?,
                w.denominator.try_view_mut()?,
                &self.options.execution.with_execution_policy(w.policy),
                w.gemm,
                w.kernels,
            )?;
        } else {
            w.denominator.as_mut_slice().fill(S::ZERO);
        }
        for (v, &r) in w
            .denominator
            .as_mut_slice()
            .iter_mut()
            .zip(self.r.as_slice())
        {
            *v = crate::checked::add(*v, r)?;
        }
        stage(solve)?;
        let factor = crate::decompose::factor_kernel(
            w.denominator.try_view()?,
            &mut *w.factor,
            &mut *w.indices,
            w.gemm,
            w.policy,
            w.kernels,
        )?;
        stage(solve)?;
        factor.solve_matrix(w.rhs.try_view()?, w.gain.as_mut_slice(), w.gemm, w.kernels)
    }

    #[expect(
        clippy::type_complexity,
        reason = "the tuple groups the distinct numerical outputs of this operation"
    )]
    fn residual(
        &self,
        w: &mut Staging<'_, '_, '_, S>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<(Scaled<S::Real>, Scaled<S::Real>), SciOxideError> {
        let n = self.a.rows();
        let m = self.b.cols();
        w.gemm.with_scaled(|mut scaled, _, gemm| {
            let first_wide = take_mut(&mut scaled, blocked::ELEMENTS)?;
            let second_wide = take_mut(&mut scaled, blocked::ELEMENTS)?;
            let feedback_wide = take_mut(&mut scaled, blocked::ELEMENTS)?;
            let (first_values, rest) =
                crate::workspace::split_mut(w.residual_values, blocked::ELEMENTS)?;
            let (second_values, feedback_values) =
                crate::workspace::split_mut(rest, blocked::ELEMENTS)?;
            let mut products = Products {
                packing: w.packing,
                workspace: gemm,
                kernels: w.kernels,
                policy: ExecutionPolicy::Sequential,
            };
            let mut error = Scaled::ZERO;
            let mut norms = [Scaled::ZERO; 4];
            let mut admit = || stage(solve);
            for r in (0..n).step_by(blocked::TILE) {
                for c in (0..n).step_by(blocked::TILE) {
                    let rows = (n - r).min(blocked::TILE);
                    let cols = (n - c).min(blocked::TILE);
                    let row_cuts = [0, rows];
                    let col_cuts = [0, cols];
                    let grid = Grid {
                        rows: &row_cuts,
                        cols: &col_cuts,
                    };
                    let mut scales = [0; 3];
                    let (s1, rest) = crate::workspace::split_mut(&mut scales, 1)?;
                    let (s2, s3) = crate::workspace::split_mut(rest, 1)?;
                    let mut first = Framed::new(
                        &mut first_values[..rows * cols],
                        &mut first_wide[..rows * cols],
                        s1,
                        grid,
                    );
                    let mut second = Framed::new(
                        &mut second_values[..rows * cols],
                        &mut second_wide[..rows * cols],
                        s2,
                        grid,
                    );
                    let mut feedback = Framed::new(
                        &mut feedback_values[..rows * cols],
                        &mut feedback_wide[..rows * cols],
                        s3,
                        grid,
                    );
                    products.product(
                        Input::Plain(self.a.view().adjoint().block(r, 0, rows, n)?),
                        Input::Plain(
                            if self.kind == RiccatiKind::Continuous {
                                w.xe.try_view()?
                            } else {
                                w.xa.try_view()?
                            }
                            .block(0, c, n, cols)?,
                        ),
                        &mut first,
                        &mut admit,
                    )?;
                    products.product(
                        Input::Plain(self.e.view().adjoint().block(r, 0, rows, n)?),
                        Input::Plain(
                            if self.kind == RiccatiKind::Continuous {
                                w.xa.try_view()?
                            } else {
                                w.xe.try_view()?
                            }
                            .block(0, c, n, cols)?,
                        ),
                        &mut second,
                        &mut admit,
                    )?;
                    products.product(
                        Input::Plain(w.rhs.try_view()?.adjoint().block(r, 0, rows, m)?),
                        Input::Plain(w.gain.try_view()?.block(0, c, m, cols)?),
                        &mut feedback,
                        &mut admit,
                    )?;
                    let (e, terms) = blocked::residual_tile(
                        [
                            first.input()?,
                            second.input()?,
                            feedback.input()?,
                            Input::Plain(self.q.view().block(r, c, rows, cols)?),
                        ],
                        [
                            S::ONE,
                            if self.kind == RiccatiKind::Continuous {
                                S::ONE
                            } else {
                                -S::ONE
                            },
                            -S::ONE,
                            S::ONE,
                        ],
                        products.packing,
                    )?;
                    error = error.add(e)?;
                    for (sum, value) in norms.iter_mut().zip(terms) {
                        *sum = sum.add(value)?;
                    }
                }
            }
            Ok((
                error.sqrt()?,
                norms
                    .into_iter()
                    .try_fold(Scaled::ZERO, |sum, value| sum.add(value.sqrt()?))?,
            ))
        })
    }
}

/// Solve staging carved from the shared workspace, with the capabilities
/// and product scratch every recovery stage uses.
struct Staging<'a, 'g, 'k, S: NumericScalar> {
    x: ScratchMatrix<'a, S>,
    xe: ScratchMatrix<'a, S>,
    xa: ScratchMatrix<'a, S>,
    xb: ScratchMatrix<'a, S>,
    rhs: ScratchMatrix<'a, S>,
    denominator: ScratchMatrix<'a, S>,
    gain: ScratchMatrix<'a, S>,
    factor: &'a mut [S],
    reduction: &'a mut [S],
    indices: &'a mut [usize],
    values: &'a mut [S::Real],
    packing: &'a mut [S],
    residual_values: &'a mut [S],
    gemm: &'a mut LinalgWorkspace<'g, S>,
    policy: ExecutionPolicy,
    kernels: KernelContext<'k>,
}

fn carve_matrix<'a, S: NumericScalar>(
    rest: &mut &'a mut [S],
    rows: usize,
    cols: usize,
) -> Result<ScratchMatrix<'a, S>, SciOxideError> {
    ScratchMatrix::new(take_mut(rest, rows * cols)?, rows, cols)
}

/// Shared-arena requirements of a solve with `n` states and `m` controls.
///
/// The direct scalar region stages X, `X E`, `X A`, `X B`, the gain
/// right-hand side and denominator, its LU factor, the gain and the
/// optional closed-loop QZ reduction scratch; the closed loop reuses `X E`/`X A`.
/// Independent verification also reserves homogeneous eigenvalue storage.
/// The index arena holds gain pivots. Product regions cover recovery, feedback
/// and triangular solves; bounded extended tiles stage residual terms.
fn requirements<S: NumericScalar>(
    (n, m): (usize, usize),
    execution: &LinalgExecutionOptions,
    policy: ExecutionPolicy,
    context: Option<KernelContext<'_>>,
    closed_loop_verification: Estimation,
) -> Result<LinalgRequirements, SciOxideError> {
    let products = execution.with_execution_policy(policy);
    let mut requirements = blocked::packing_requirements::<S>(execution, context)?;
    for (rows, shared, cols) in [
        (n, n, n),
        (n, n, m),
        (m, n, n),
        (m, n, m),
        (n, m, n),
        (m, m, m),
        (m, m, n),
    ] {
        let mut request = DenseWorkspaceRequest::product(rows, shared, cols, &products);
        if let Some(context) = context {
            request = request.for_kernels(context);
        }
        requirements = requirements.max(&crate::kernel::product_requirements::<S>(&request)?);
    }
    let size =
        |shape: &[usize]| checked_product(shape, scioxide_core::quantity!("riccati_scratch"));
    let factor = crate::decompose::factor_kernel_lengths(m)?;
    *requirements.scalar_mut(ScalarRegion::Direct) = [
        size(&[3, n, n])?,
        size(&[4, n, m])?,
        size(&[m, m])?,
        factor.scalars,
        if closed_loop_verification == Estimation::Estimate {
            crate::eigen::qz_kernel::scratch_len::<S>(n, policy)?
        } else {
            0
        },
        blocked::PACKING + 3 * blocked::ELEMENTS,
    ]
    .into_iter()
    .try_fold(0usize, |total, part| {
        checked_add(total, part, scioxide_core::quantity!("riccati_scratch"))
    })?;
    *requirements.real_mut(RealRegion::Direct) = if closed_loop_verification == Estimation::Estimate
    {
        size(&[3, n])?
    } else {
        0
    };
    requirements.scaled = 3 * blocked::ELEMENTS;
    requirements.indices = factor.indices;
    requirements.byte_limit = execution.byte_limit();
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

fn identity(n: usize, m: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "riccati", 1),
        ((n as u128) << 64) | m as u128,
        0,
    )
}

/// Count one recovery or preparation stage against the iteration budget.
fn stage(solve: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    crate::solve_support::admit_iteration(
        Some(solve.execution()),
        Some(solve),
        "solve iteration budget exhausted",
    )
}

fn validate_problem_shape<S: NumericScalar>(
    problem: &RiccatiProblem<'_, S>,
) -> Result<(usize, usize), SciOxideError> {
    let n = problem.a.rows();
    let m = problem.b.cols();
    for v in [problem.a, problem.q] {
        crate::checked::validate_matrix(v, n, n)?;
    }
    crate::checked::validate_matrix(problem.b, n, m)?;
    crate::checked::validate_matrix(problem.r, m, m)?;
    hermitian(problem.q, scioxide_core::argument!("q"))?;
    hermitian(problem.r, scioxide_core::argument!("r"))?;
    Ok((n, m))
}

/// Positive-cost scalar CARE/DARE, avoiding an arbitrarily small U1.
/// Absence means the equation requires the general ordered-pencil algorithm.
fn scalar_solution<S: NumericScalar>(
    coefficients: &RiccatiCoefficients<S>,
    kind: RiccatiKind,
    options: &RiccatiOptions<S::Real>,
) -> Result<Option<S::Real>, SciOxideError> {
    let RiccatiCoefficients { a, b, q, r, e, s } = coefficients;
    if a.rows() != 1 || b.cols() != 1 || e.as_slice()[0] != S::ONE || s.as_slice()[0] != S::ZERO {
        return Ok(None);
    }
    let (a, b) = (a.as_slice()[0], b.as_slice()[0]);
    let (q, r) = (
        q.as_slice()[0].real_component(),
        r.as_slice()[0].real_component(),
    );
    if q <= S::Real::ZERO || r <= S::Real::ZERO || b == S::ZERO {
        return Ok(None);
    }
    let ar = Scaled::from_value(a.real_component())?;
    let ai = Scaled::from_value(a.imaginary_component())?;
    let bb = ScaledValue::from_value(b)?.magnitude()?.square()?;
    let qr = Scaled::from_value(q)?.mul(Scaled::from_value(r)?)?;
    // Accumulate the polynomial's linear coefficient before rounding: in DARE
    // r*(1-|a|^2)-q*|b|^2 can contain cancellation at the unit circle.
    let mut linear = scioxide_core::RangeAccumulator::<S::Real>::new();
    if kind == RiccatiKind::Continuous {
        linear.add_product(-r, a.real_component())?;
        linear.add_product(-r, a.real_component())?;
    } else {
        linear.add(r)?;
        for part in [a.real_component(), a.imaginary_component()] {
            linear.add_triple_product(-r, part, part)?;
        }
        for part in [b.real_component(), b.imaginary_component()] {
            linear.add_triple_product(-q, part, part)?;
        }
    }
    let linear = linear.finish_scaled()?;
    // bb*x^2 + linear*x - qr = 0 has exactly one positive root. Use the
    // same-sign quadratic formula, recovering the other branch by the root
    // product when linear >= 0. Keeping coefficients and roots extended avoids
    // losing bb when a common IEEE coefficient scale would underflow it.
    let radical = linear.square()?.add(bb.mul(qr)?.shifted(2)?)?.sqrt()?;
    let numerator = linear.abs().add(radical)?.shifted(-1)?;
    let value = if linear.is_sign_negative() {
        numerator.div(bb)?
    } else {
        qr.div(numerator)?
    }
    .round()?;
    if value <= S::Real::ZERO {
        return Err(instability(
            "scalar Riccati equation has no positive stabilizing root",
        ));
    }
    let g = bb.div(Scaled::from_value(r)?)?;
    let gap = if kind == RiccatiKind::Continuous {
        let real = ar.square()?.add(g.mul(Scaled::from_value(q)?)?)?.sqrt()?;
        let magnitude = real.square()?.add(ai.square()?)?.sqrt()?;
        real.div(if magnitude.abs_le(Scaled::ONE) {
            Scaled::ONE
        } else {
            magnitude
        })?
    } else {
        // Closed-loop eigenvalue is a/(1+g*x); this avoids cancellation in a-b*k.
        let denominator = Scaled::ONE.add(g.mul(Scaled::from_value(value)?)?)?;
        let magnitude = ScaledValue::from_value(a)?.magnitude()?;
        denominator.sub(magnitude)?.div(denominator)?
    };
    if gap.round()? <= options.stability_tolerance.threshold(S::Real::ONE)? {
        return Err(instability(
            "pencil spectrum is ambiguous at the stability boundary",
        ));
    }
    Ok(Some(value))
}

/// Scale-only Parlett–Reinsch balance followed by Benner's symplectic projection.
/// The returned D satisfies x = D*z; pencil similarity uses diag(D^-1,D,C).
fn balance<S: NumericScalar>(
    h: &mut Matrix<S>,
    j: &mut Matrix<S>,
    n: usize,
    balancing: RiccatiBalancing,
    solve: &mut SolveContext<'_, '_>,
) -> Result<Vec<S::Real>, SciOxideError> {
    let mut scales = try_filled(n, S::Real::ONE, scioxide_core::quantity!("scales"))?;
    if balancing == RiccatiBalancing::None {
        return Ok(scales);
    }
    let order = h.rows();
    let mut magnitudes = try_filled(
        h.as_slice().len(),
        Scaled::ZERO,
        scioxide_core::quantity!("balance"),
    )?;
    for (position, value) in magnitudes.iter_mut().enumerate() {
        if position / order != position % order {
            *value = ScaledValue::from_value(h.as_slice()[position])?
                .magnitude()?
                .add(ScaledValue::from_value(j.as_slice()[position])?.magnitude()?)?;
        }
    }
    let mut exponents = try_filled(order, 0_i64, scioxide_core::quantity!("balance"))?;
    let reduction = Scaled::from_value(S::Real::try_from_f64_lossy(0.95)?)?;
    loop {
        solve.execution().check()?;
        let mut changed = false;
        for (index, exponent) in exponents.iter_mut().enumerate() {
            let mut row = Scaled::ZERO;
            let mut column = Scaled::ZERO;
            for k in 0..order {
                row = row.add(magnitudes[index * order + k])?;
                column = column.add(magnitudes[k * order + index])?;
            }
            if row.is_zero() || column.is_zero() {
                continue;
            }
            let mut shift = (row.exponent() - column.exponent()).div_euclid(2);
            let mut r = row.shifted(-shift)?;
            let mut c = column.shifted(shift)?;
            if c.abs_le(r.shifted(-1)?) {
                shift += 1;
                r = r.shifted(-1)?;
                c = c.shifted(1)?;
            } else if r.abs_le(c.shifted(-1)?) {
                shift -= 1;
                r = r.shifted(1)?;
                c = c.shifted(-1)?;
            }
            if shift != 0 && r.add(c)?.abs_le(row.add(column)?.mul(reduction)?) {
                *exponent += shift;
                changed = true;
                for k in 0..order {
                    magnitudes[index * order + k] =
                        magnitudes[index * order + k].shifted(-shift)?;
                    magnitudes[k * order + index] = magnitudes[k * order + index].shifted(shift)?;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (i, scale) in scales.iter_mut().enumerate() {
        // Round half-integer exponents to even, following Benner's balancing method.
        let difference = exponents[n + i] - exponents[i];
        let lower = difference.div_euclid(2);
        let exponent = lower + i64::from(difference.rem_euclid(2) != 0 && lower % 2 != 0);
        exponents[i] = exponent;
        exponents[n + i] = -exponent;
        *scale = Scaled::ONE.shifted(-exponent)?.try_value()?;
    }
    // The loop accumulates row divisors; applying row-minus-column exponents
    // needs row multipliers. The state/costate projection above already converts
    // that convention, while retained control exponents still need negation.
    for exponent in &mut exponents[2 * n..] {
        *exponent = -*exponent;
    }
    for matrix in [h, j] {
        for (position, value) in matrix.as_mut_slice().iter_mut().enumerate() {
            *value = ScaledValue::from_value(*value)?
                .shifted(exponents[position / order] - exponents[position % order])?
                .round()?;
        }
    }
    Ok(scales)
}

fn extended_pencil<S: NumericScalar>(
    coefficients: &RiccatiCoefficients<S>,
    kind: RiccatiKind,
    dimensions: (usize, usize),
) -> Result<(Matrix<S>, Matrix<S>), SciOxideError> {
    let RiccatiCoefficients { a, b, q, r, e, s } = coefficients;
    let n = a.rows();
    let m = b.cols();
    let (nn, total) = dimensions;
    let mut h = Matrix::zeros(total, total)?;
    let mut j = Matrix::zeros(total, total)?;
    for row in 0..n {
        for col in 0..n {
            let av = a.get_element(row, col);
            let ev = e.get_element(row, col);
            let qv = q.get_element(row, col);
            h.as_mut_slice()[row * total + col] = av;
            h.as_mut_slice()[(n + row) * total + col] = -qv;
            j.as_mut_slice()[row * total + col] = ev;
            if kind == RiccatiKind::Continuous {
                h.as_mut_slice()[(n + col) * total + n + row] = -av.conj();
                j.as_mut_slice()[(n + col) * total + n + row] = ev.conj();
            } else {
                h.as_mut_slice()[(n + col) * total + n + row] = ev.conj();
                j.as_mut_slice()[(n + col) * total + n + row] = av.conj();
            }
        }
    }
    for row in 0..n {
        for col in 0..m {
            let bv = b.get_element(row, col);
            let sv = s.get_element(row, col);
            h.as_mut_slice()[row * total + nn + col] = bv;
            h.as_mut_slice()[(n + row) * total + nn + col] = -sv;
            h.as_mut_slice()[(nn + col) * total + row] = sv.conj();
            if kind == RiccatiKind::Continuous {
                h.as_mut_slice()[(nn + col) * total + n + row] = bv.conj();
            } else {
                j.as_mut_slice()[(nn + col) * total + n + row] = -bv.conj();
            }
        }
    }
    for row in 0..m {
        for col in 0..m {
            h.as_mut_slice()[(nn + row) * total + nn + col] = r.get_element(row, col);
        }
    }
    Ok((h, j))
}

/// Deflate the `m` control columns of the extended pencil.
///
/// With the pivoted QR `[B; -S; R] = Q [R_c; 0]`, the last `2n` columns of Q
/// span the orthogonal complement, and compressing both pencil matrices from
/// the left with it removes the control unknowns, leaving a `2n x 2n` pencil
/// whose finite spectrum is that of the Hamiltonian/symplectic problem.
fn deflate_control<'c, S: NumericScalar>(
    (h, j): (&Matrix<S>, &Matrix<S>),
    m: usize,
    decomposition: &LinalgExecutionOptions,
    (workspace, kernels, solve): (
        &mut LinalgWorkspace<'_, S>,
        KernelContext<'c>,
        &mut SolveContext<'_, 'c>,
    ),
) -> Result<(Matrix<S>, Matrix<S>), SciOxideError> {
    let total = h.rows();
    let nn = total - m;
    stage(solve)?;
    let mut qr = HouseholderDecomposition::empty();
    HouseholderDecomposition::factor_into(
        h.view().block(0, nn, total, m)?,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
        decomposition,
        &mut qr,
        workspace,
        kernels.scoped(&mut *solve),
    )?;
    let mut q = Matrix::zeros(total, total)?;
    qr.q_matrix_into(
        decomposition,
        q.view_mut(),
        workspace,
        kernels.scoped(&mut *solve),
    )?;
    let complement = q.view().block(0, m, total, nn)?;
    let mut compressed = [Matrix::zeros(nn, nn)?, Matrix::zeros(nn, nn)?];
    for (source, target) in [h, j].into_iter().zip(&mut compressed) {
        stage(solve)?;
        complement.adjoint().mul_mat_into(
            source.view().block(0, 0, total, nn)?,
            decomposition,
            target.view_mut(),
            workspace,
            kernels.scoped(&mut *solve),
        )?;
    }
    let [hp, jp] = compressed;
    Ok((hp, jp))
}

/// Factor the deflated pencil with the arena QZ kernel, select its stable
/// half (left half plane for CARE, inside the unit circle for DARE), reorder
/// it to the front with condition estimates and extract the graph data.
///
/// Eigenvalues too close to the boundary make the stabilizing solution
/// ill-defined, and a count other than n means no stabilizing solution
/// exists; both fail explicitly rather than returning an unreliable X.
fn stable_graph<'c, S: NumericScalar>(
    (hp, jp): (&Matrix<S>, &Matrix<S>),
    (e, scales): (Option<&Matrix<S>>, &[S::Real]),
    kind: RiccatiKind,
    (options, policy): (&RiccatiOptions<S::Real>, ExecutionPolicy),
    workspace: &mut LinalgWorkspace<'_, S>,
    kernels: KernelContext<'c>,
    solve: &mut SolveContext<'_, 'c>,
) -> Result<StableGraph<S>, SciOxideError> {
    let n = scales.len();
    let nn = hp.rows();
    let qz_options = SchurOptions::new().with_execution_options(
        options
            .execution
            .with_execution_policy(policy)
            .with_iteration_budget(options.stage_iteration_budget()),
    );
    let reorder_options = SchurReorderOptions::new()
        .with_execution_options(
            options
                .execution
                .with_execution_policy(policy)
                .with_iteration_budget(options.stage_iteration_budget()),
        )
        .with_condition_estimates(Estimation::Skip);
    let mut requirements = GeneralizedSchurDecomposition::<S>::workspace_requirements(
        &GeneralizedSchurWorkspaceRequest::new(nn, &qz_options),
    )?
    .max(&GeneralizedSchurDecomposition::<S>::workspace_requirements(
        &GeneralizedSchurWorkspaceRequest::reorder(nn, &reorder_options),
    )?);
    requirements.byte_limit = options.execution.byte_limit();
    workspace.try_reserve(&requirements)?;
    let mut mask = try_filled(nn, false, scioxide_core::quantity!("mask"))?;
    let mut values = Vec::new();
    scioxide_core::backend::try_reserve(
        &mut values,
        nn,
        scioxide_core::quantity!("pencil_eigenvalues"),
    )?;
    let mut graph = StableGraph {
        denominator: Matrix::zeros(n, n)?,
        leading: Matrix::zeros(n, n)?,
        rhs: Matrix::zeros(n, n)?,
    };
    workspace.with_direct(|mut scalars, _, indices, child| {
        let mut pencil = Pencil {
            s: take_square(&mut scalars, nn)?,
            t: take_square(&mut scalars, nn)?,
            q: Some(take_square(&mut scalars, nn)?),
            z: Some(take_square(&mut scalars, nn)?),
        };
        pencil.s.as_mut_slice().copy_from_slice(hp.as_slice());
        pencil.t.as_mut_slice().copy_from_slice(jp.as_slice());
        crate::eigen::qz_pencil(&mut pencil, scalars, &qz_options, solve)?;
        crate::eigen::for_each_generalized_value(&pencil.s, &pencil.t, |_, block| {
            values.extend_from_slice(block)
        })?;
        let threshold = options.stability_tolerance.threshold(S::Real::ONE)?;
        for (selected, &value) in mask.iter_mut().zip(&values) {
            let gap = margin(value, kind)?;
            if gap.abs() <= threshold {
                return Err(instability(
                    "pencil spectrum is ambiguous at the stability boundary",
                ));
            }
            *selected = gap > S::Real::ZERO;
        }
        if mask.iter().filter(|&&x| x).count() != n {
            return Err(instability(
                "pencil does not have the required stabilizing subspace dimension",
            ));
        }
        crate::eigen::qz_reorder(
            &mut pencil,
            &mask,
            indices,
            child.extended_mut(),
            &reorder_options,
            solve,
        )?;
        let Some(u) = pencil.z.as_ref() else {
            return Err(instability("deflating subspace is unavailable"));
        };
        for i in 0..n {
            for col in 0..n {
                graph.leading.as_mut_slice()[i * n + col] = u.get_element(i, col);
                graph.rhs.as_mut_slice()[col * n + i] = u.get_element(n + i, col).conj();
            }
        }
        Ok(())
    })?;
    // E' U1, with E'=D^-1 E D; X'=U2 (E' U1)^-1. Extract U1 before
    // releasing the QZ arena, then use the shared range-safe Standard GEMM.
    if let Some(e) = e {
        stage(solve)?;
        let mut balanced = e.view().try_to_matrix()?;
        for (position, value) in balanced.as_mut_slice().iter_mut().enumerate() {
            *value = scale(*value, scales[position % n], scales[position / n])?;
        }
        graph.leading.view().adjoint().mul_mat_into(
            balanced.view().adjoint(),
            &options
                .execution
                .with_execution_policy(policy)
                .with_accuracy(scioxide_core::Accuracy::Standard),
            graph.denominator.view_mut(),
            workspace,
            kernels.scoped(&mut *solve),
        )?;
    } else {
        for i in 0..n {
            for col in 0..n {
                graph.denominator.as_mut_slice()[col * n + i] =
                    graph.leading.get_element(i, col).conj();
            }
        }
    }
    Ok(graph)
}

fn hermitian<S: NumericScalar>(
    a: MatrixView<'_, S>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    for i in 0..a.rows() {
        for j in i..a.cols() {
            if a.get_element(i, j) != a.get_element(j, i).conj() {
                return Err(SciOxideError::invalid_input(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "cost matrix must be exactly Hermitian",
                ));
            }
        }
    }
    Ok(())
}
fn instability(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::IllConditioned, detail)
}
fn scale<S: NumericScalar>(x: S, num: S::Real, den: S::Real) -> Result<S, SciOxideError> {
    scalar(
        extended(x)?
            .mul(ScaledValue::from_value(scioxide_core::Complex::new(
                num,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))?)?
            .div(ScaledValue::from_value(scioxide_core::Complex::new(
                den,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))?)?,
    )
}
fn margin<R: RealScalar>(
    v: GeneralizedEigenvalue<R>,
    kind: RiccatiKind,
) -> Result<R, SciOxideError> {
    if v.is_infinite() {
        return Ok(-R::ONE);
    }
    let a = v.alpha();
    let b = v.beta();
    let common = a.re.abs().max(a.im.abs()).max(b);
    if common == R::ZERO {
        return Err(instability("singular homogeneous eigenvalue"));
    }
    let real = a.re / common;
    let imaginary = a.im / common;
    let beta = b / common;
    let magnitude = real.hypot(imaginary);
    let denominator = magnitude.max(beta);
    Ok(if kind == RiccatiKind::Continuous {
        -real / denominator
    } else {
        (beta - magnitude) / denominator
    })
}
/// Dense recovery and residual product under the solve's execution options;
/// product kernels start no solve scope.
fn multiply<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    out: MatrixViewMut<'_, S>,
    options: &LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    kernels: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    a.mul_mat_into(
        b,
        options,
        out,
        workspace,
        crate::LinalgExecutionContext::unscoped(kernels),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn extended_pencil_retains_control_exponents_as_row_multipliers() -> Result<(), SciOxideError> {
        for kind in [RiccatiKind::Continuous, RiccatiKind::Discrete] {
            // These powers give exact loop exponents [-52, 28, -12] and
            // [54, -26, 14], respectively. Projection makes the state/costate
            // exponents reciprocal; the controls must negate the loop exponent.
            for (sign, control_exponent) in [(-1, 12), (1, -14)] {
                let coefficients = RiccatiCoefficients {
                    a: Matrix::zeros(1, 1)?,
                    b: Matrix::from_vec(1, 1, alloc::vec![2_f64.powi(40 * sign)])?,
                    q: Matrix::from_vec(1, 1, alloc::vec![2_f64.powi(-80 * sign)])?,
                    r: Matrix::from_vec(1, 1, alloc::vec![2_f64.powi(-40 * sign)])?,
                    e: Matrix::identity(1)?,
                    s: Matrix::zeros(1, 1)?,
                };
                let (mut h, mut j) = extended_pencil(&coefficients, kind, (2, 3))?;
                let mut state = scioxide_core::SolveState::new();
                let mut budgets = scioxide_core::SolveBudgets::unbounded();
                let mut solve = state.begin_solve(
                    ExecutionContext::sequential(),
                    identity(1, 1),
                    &mut budgets,
                )?;
                let scales = balance(&mut h, &mut j, 1, RiccatiBalancing::Scale, &mut solve)?;
                assert_eq!(scales, alloc::vec![2_f64.powi(40 * sign)]);
                assert!((h.get_element(1, 0) + 1.).abs() <= f64::EPSILON);
                let input = h.get_element(0, 2);
                let control = if kind == RiccatiKind::Continuous {
                    h.get_element(2, 1)
                } else {
                    -j.get_element(2, 1)
                };
                assert!((input / 2_f64.powi(-control_exponent) - 1.).abs() <= f64::EPSILON);
                assert!((control / 2_f64.powi(control_exponent) - 1.).abs() <= f64::EPSILON);
                assert!(
                    h.as_slice()
                        .iter()
                        .chain(j.as_slice())
                        .all(|x| x.is_finite())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn control_similarity_reduces_asymmetric_magnitude_ratios() -> Result<(), SciOxideError> {
        // Isolate the control similarity from the state/costate projection.
        // This asymmetric kernel fixture exercises both signs without imposing
        // the Hermitian coefficient constraints of the public Riccati API.
        for (upper, lower) in [(2_f64.powi(80), 1.), (1., 2_f64.powi(80))] {
            let mut h = Matrix::from_vec(
                4,
                4,
                alloc::vec![
                    0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., upper, 0., 0., lower, 1.,
                ],
            )?;
            let mut j = Matrix::zeros(4, 4)?;
            let mut state = scioxide_core::SolveState::new();
            let mut budgets = scioxide_core::SolveBudgets::unbounded();
            let mut solve =
                state.begin_solve(ExecutionContext::sequential(), identity(1, 2), &mut budgets)?;
            let scales = balance(&mut h, &mut j, 1, RiccatiBalancing::Scale, &mut solve)?;
            assert_eq!(scales, alloc::vec![1.]);
            let ratio = h.get_element(2, 3) / h.get_element(3, 2);
            assert!((ratio - 1.).abs() <= f64::EPSILON);
            assert!(ratio.max(1. / ratio) < (upper / lower).max(lower / upper));
        }
        Ok(())
    }

    fn homogeneous<R: RealScalar>()
    where
        scioxide_core::Complex<R>: NumericScalar<Real = R>,
    {
        let big = R::try_from_f64_lossy(if core::mem::size_of::<R>() == 4 {
            3e38
        } else {
            1.5e308
        })
        .unwrap();
        let beta = R::try_from_f64_lossy(if core::mem::size_of::<R>() == 4 {
            2e38
        } else {
            1e308
        })
        .unwrap();
        let e = Matrix::from_vec(1, 1, vec![scioxide_core::Complex::new(beta, R::ZERO)]).unwrap();
        for sign in [R::ONE, -R::ONE] {
            let a =
                Matrix::from_vec(1, 1, vec![scioxide_core::Complex::new(sign * big, big)]).unwrap();
            let factor = GeneralizedSchurDecomposition::decompose(&a, &e).unwrap();
            let value = factor.eigenvalues()[0];
            let continuous = margin(value, RiccatiKind::Continuous).unwrap();
            let discrete = margin(value, RiccatiKind::Discrete).unwrap();
            assert!(continuous.is_finite() && discrete.is_finite());
            let reciprocal_root = R::try_from_f64_lossy(core::f64::consts::FRAC_1_SQRT_2).unwrap();
            let tolerance = R::try_from_usize_exact(512).unwrap() * R::EPSILON;
            assert!((continuous + sign * reciprocal_root).abs() < tolerance);
            assert!((discrete - (beta / big * reciprocal_root - R::ONE)).abs() < tolerance);
        }
    }
    #[test]
    fn homogeneous32() {
        homogeneous::<f32>()
    }
    #[test]
    fn homogeneous64() {
        homogeneous::<f64>()
    }
    #[test]
    fn requirements_cover_context_and_overflow() {
        let options = LinalgExecutionOptions::new();
        let default = requirements::<f64>(
            (3, 2),
            &options,
            ExecutionPolicy::Auto,
            None,
            Estimation::Skip,
        )
        .unwrap();
        let sequential = requirements::<f64>(
            (3, 2),
            &options,
            ExecutionPolicy::Sequential,
            Some(KernelContext::sequential()),
            Estimation::Skip,
        )
        .unwrap();
        assert_eq!(default.max(&sequential), default);
        assert_eq!(
            default.real(crate::workspace::RealRegion::Exact)
                + default.real(crate::workspace::RealRegion::Direct),
            0
        );
        assert_eq!(default.indices, 6);
        let verified = requirements::<f64>(
            (3, 2),
            &options,
            ExecutionPolicy::Auto,
            None,
            Estimation::Estimate,
        )
        .unwrap();
        assert!(verified.scalar_len().unwrap() > default.scalar_len().unwrap());
        assert_eq!(
            verified.real(crate::workspace::RealRegion::Exact)
                + verified.real(crate::workspace::RealRegion::Direct),
            9
        );
        assert_eq!(
            requirements::<f64>(
                (usize::MAX, 1),
                &options,
                ExecutionPolicy::Auto,
                None,
                Estimation::Skip
            )
            .unwrap_err()
            .kind(),
            scioxide_core::SciOxideErrorKind::SizeOverflow
        );
    }

    #[test]
    fn optional_verification_rejects_the_antistabilizing_branch() -> Result<(), SciOxideError> {
        fn assert_rejected(
            plan: &RiccatiPlan<f64>,
            termination: DirectSolveTermination,
        ) -> Result<(), SciOxideError> {
            let result = plan.solve()?;
            assert_eq!(result.report().termination(), termination);
            assert!(result.report().work() > 0);
            assert!(result.solution().is_none());
            assert!(result.gain().is_none());
            let (values, report) = result.into_parts();
            assert!(values.is_none());
            assert_eq!(report.termination(), termination);
            let mut x = Matrix::from_vec(1, 1, alloc::vec![71.])?;
            let mut gain = Matrix::from_vec(1, 1, alloc::vec![72.])?;
            let mut workspace = plan.workspace(())?;
            let report = plan.solve_into(
                x.view_mut(),
                gain.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )?;
            assert_eq!(report.termination(), termination);
            assert_eq!(
                report.work(),
                workspace
                    .last_attempt()
                    .unwrap()
                    .counts()
                    .total()?
                    .iterations()
            );
            assert_eq!(x.as_slice(), &[71.]);
            assert_eq!(gain.as_slice(), &[72.]);
            Ok(())
        }
        let a = Matrix::identity(1)?;
        let problem = RiccatiProblem {
            a: a.view(),
            b: a.view(),
            q: a.view(),
            r: a.view(),
            e: None,
            s: None,
        };
        let mut plan =
            RiccatiPlan::<f64>::new(problem, RiccatiKind::Continuous, RiccatiOptions::new())?;
        // Inject the other exact CARE branch into scalar recovery to test
        // independent verification, without changing the original equation.
        plan.recovery = Recovery::Scalar(1. - 2.0_f64.sqrt());
        let residual_only = plan.solve()?;
        assert_eq!(
            residual_only.report.termination,
            DirectSolveTermination::Accepted
        );
        assert!(residual_only.report.stability_margin.is_none());
        assert!(residual_only.solution().is_some());
        assert!(residual_only.gain().is_some());
        assert!(residual_only.into_parts().0.is_some());
        plan.options.closed_loop_verification = Estimation::Estimate;
        let verified = plan.solve()?;
        assert_eq!(
            verified.report.termination,
            DirectSolveTermination::StabilityRejected
        );
        assert!(
            verified
                .report
                .stability_margin
                .is_some_and(|margin| margin < 0.)
        );
        assert_rejected(&plan, DirectSolveTermination::StabilityRejected)?;
        // A finite symmetric candidate with a large residual must still be
        // rejected when the independent spectrum check is disabled.
        plan.recovery = Recovery::Scalar(2. * (1. - 2.0_f64.sqrt()));
        plan.options.closed_loop_verification = Estimation::Skip;
        assert_eq!(
            plan.solve()?.report.termination,
            DirectSolveTermination::ResidualRejected
        );
        assert_rejected(&plan, DirectSolveTermination::ResidualRejected)?;
        plan.options.closed_loop_verification = Estimation::Estimate;
        // Both checks fail: independent stability rejection has precedence.
        assert_rejected(&plan, DirectSolveTermination::StabilityRejected)?;
        Ok(())
    }
}
