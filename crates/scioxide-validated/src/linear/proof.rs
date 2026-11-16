//! The verified linear-system verifier.
use super::workspace::{LaneView, ProofScratch, Shape, SweepScratch};
use super::*;
use scioxide_core::SolveContext;
use scioxide_core::{
    AlgebraicScalar, AlgorithmId, BlockVectorView, BlockVectorViewMut, EvaluationRole,
    ExecutionContext, Invocation, OperatorContext, PartialResultTrust, ProblemToken, SolveIdentity,
    SolveScope,
};

/// Completed matrix proof borrowed only within the call that computed it.
#[derive(Clone, Copy)]
struct Regularity<'a, R: RealScalar> {
    report: LinearVerificationReport<R>,
    rows: &'a [R],
}
pub(crate) const ALGORITHM: AlgorithmId =
    scioxide_core::algorithm_id!("validated-linear", "streamed-krawczyk", 1);

/// Proves that a square linear system `Ax = b` is nonsingular and bounds the
/// error of an approximate solution.
///
/// The system matrix `A` and the preconditioner `R`, an approximate inverse of
/// `A`, are [`IntervalLinearOperator`]s. They may stand for sets of matrices,
/// for example matrices with interval entries, and the right-hand side `b` is
/// a vector of [`IntervalBox`]es; every statement below holds for every member.
/// `R` usually comes from an LU factorization through [`LuIntervalInverse`];
/// the verifier does not trust it.
///
/// # Method
///
/// For a center `x̃`, a proof encloses the correction `z ∋ R(b − Ax̃)` and
/// bounds each row of `|I − RA|`, `rowᵢ`, in the norm of [`LinearBounds`]. If
/// `q = maxᵢ rowᵢ < 1`, then `RA` and therefore `A` are nonsingular, and the
/// error `e = x* − x̃` of the solution `x*` satisfies `e = R(b − Ax̃) + (I − RA)e`.
/// This gives the contraction bound
///
/// ```text
/// ‖e‖ ≤ ρ = ‖z‖ / (1 − q),    |eᵢ| ≤ |zᵢ| + rowᵢ·ρ,
/// ```
///
/// applied separately to the real and imaginary parts and rounded upward. The
/// radii that the verifier computes are this bound times `1 + 4ε`, where `ε` is
/// the machine epsilon, so that [`Self::verify`] accepts them when it
/// recomputes `z` at the returned center. For point data, `ρ` exceeds `‖e‖`
/// by at most the factor `(1 + q)/(1 − q)`, plus rounding and the widths of the
/// enclosures. [`Self::verify`] accepts proposed radii `r` when the box is
/// mapped into itself, `|zᵢ| + (|I − RA|·r)ᵢ ≤ rᵢ` for every `i`, or when they
/// contain the contraction bound.
///
/// Every call recomputes all bounds; nothing is cached between calls. One pass
/// over the matrix serves all right-hand sides of a call. A call that is not
/// `Verified` leaves its output unchanged. The verifier checks on every call that
/// the operators still have the shape and scratch length they had at
/// construction.
///
/// | Method | Use |
/// |---|---|
/// | [`Self::solve`], [`Self::solve_into`] | compute and certify a solution |
/// | [`Self::solve_block_into`] | several right-hand sides, in parallel |
/// | [`Self::propose`], [`Self::propose_into`] | certify your own approximate solution |
/// | [`Self::verify`], [`Self::verify_scoped`] | check a proposed solution and error bound |
/// | [`Self::bounds`], [`Self::bounds_scoped`] | prove nonsingularity and bound `‖A⁻¹‖` and the condition number |
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::MatrixView;
/// use scioxide_validated::{
///     DenseIntervalOperator, LinearCertificate, ValidatedOptions, LinearVerifier, point_enclosure,
/// };
///
/// // A = [4 1; 1 3] and its exact inverse R = [3 −1; −1 4] / 11.
/// let a = [4.0_f64, 1.0, 1.0, 3.0];
/// let r = [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0];
/// let system = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &a)?)?;
/// let inverse = DenseIntervalOperator::point(MatrixView::from_slice(2, 2, &r)?)?;
/// let verifier = LinearVerifier::new(&system, &inverse, ValidatedOptions::new())?;
/// // b = (1, 2) has the solution x = (1/11, 7/11).
/// let rhs = [point_enclosure(1.0)?, point_enclosure(2.0)?];
/// let (report, certificate) = verifier.solve(&rhs)?;
/// assert!(report.is_verified());
/// // Solves compute ‖A‖ and ‖I − RA‖ only; `bounds` also bounds ‖R‖ and ‖A⁻¹‖.
/// assert!(
///     report
///         .bounds()
///         .expect("complete pass")
///         .preconditioner()
///         .is_none()
/// );
/// let certificate = certificate.expect("present when verified");
/// assert!(certificate.enclosure()?[0].re.contains(1.0 / 11.0));
/// assert!(
///     verifier.bounds()?
///         .bounds()
///         .expect("complete pass")
///         .inverse()
///         .is_some()
/// );
/// // With a workspace, later solves do not allocate.
/// let mut workspace = verifier.workspace(1)?;
/// let mut out = LinearCertificate::new(2)?;
/// let report = verifier.solve_into(
///     &rhs,
///     out.view_mut(),
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(report.is_verified());
/// assert!(verifier.verify(&rhs, out.view())?.is_verified());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Neumaier, *Interval Methods for Systems of Equations*, Cambridge
///   University Press, 1990.
/// - S. M. Rump, "Verification methods: Rigorous results using floating-point
///   arithmetic", *Acta Numerica* 19, 2010, pp. 287–449.
#[derive(Clone, Copy)]
pub struct LinearVerifier<'a, S: NumericScalar> {
    system: &'a (dyn IntervalLinearOperator<S> + Sync),
    inverse: &'a (dyn IntervalLinearOperator<S> + Sync),
    n: usize,
    a_scratch: usize,
    r_scratch: usize,
    scalar_scratch: usize,
    options: ValidatedOptions,
}
impl<S: NumericScalar> core::fmt::Debug for LinearVerifier<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LinearVerifier")
            .field("order", &self.n)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<'a, S: NumericScalar> LinearVerifier<'a, S> {
    /// Creates a verifier for the matrix `system` and the preconditioner `inverse`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if an operator is not square or the two differ in
    ///   order.
    /// - Errors of the operators' `enclosure_workspace_len`.
    pub fn new(
        system: &'a (dyn IntervalLinearOperator<S> + Sync),
        inverse: &'a (dyn IntervalLinearOperator<S> + Sync),
        options: ValidatedOptions,
    ) -> Result<Self> {
        let n = system.rows();
        crate::shell::exact(n, system.cols())?;
        crate::shell::exact(n, inverse.rows())?;
        crate::shell::exact(n, inverse.cols())?;
        Ok(Self {
            system,
            inverse,
            n,
            a_scratch: system.enclosure_workspace_len()?,
            r_scratch: inverse.enclosure_workspace_len()?,
            scalar_scratch: inverse.apply_workspace_len()?,
            options,
        })
    }
    /// Returns the order `n` of the system; it may be zero.
    pub const fn order(&self) -> usize {
        self.n
    }
    /// Returns the options.
    pub const fn options(&self) -> &ValidatedOptions {
        &self.options
    }
    fn shape(&self) -> Shape {
        Shape {
            order: self.n,
            provider: self.a_scratch.max(self.r_scratch),
            scalar: self.scalar_scratch,
        }
    }
    fn identity(&self) -> SolveIdentity {
        SolveIdentity::new(ProblemToken::new(0), ALGORITHM, self.n as u128, 0)
    }
    /// Returns the workspace size for single proofs and for block solves of up
    /// to `right_hand_sides` columns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(&self, right_hand_sides: usize) -> Result<LinearRequirements> {
        LinearRequirements::new::<S>(self.shape(), right_hand_sides, self.options.byte_limit())
    }
    /// Allocates a workspace for up to `right_hand_sides` columns.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if it cannot be sized or reserved.
    pub fn workspace(&self, right_hand_sides: usize) -> Result<LinearWorkspace<S>> {
        let mut workspace = LinearWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(right_hand_sides)?)?;
        Ok(workspace)
    }
    /// Check the environment and that the providers of `self` are unchanged.
    fn validate(&self) -> Result<()> {
        crate::environment::check()?;
        crate::protocol::validate_linear_provider(self.system, self.n, self.a_scratch)?;
        crate::protocol::validate_linear_provider(self.inverse, self.n, self.r_scratch)?;
        if self.inverse.apply_workspace_len()? != self.scalar_scratch {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "provider shape or requirements changed",
            ));
        }
        Ok(())
    }
    /// Validate the `rhs` argument.
    fn rhs(&self, rhs: &[IntervalBox<S::Real>]) -> Result<()> {
        crate::shell::exact(self.n, rhs.len())?;
        validate_boxes::<S>(rhs, true, scioxide_core::argument!("rhs"))
    }
    /// Validate the `out` argument.
    fn output(&self, out: &LinearCertificateViewMut<'_, S>) -> Result<()> {
        crate::shell::exact(self.n, out.center.len())?;
        crate::shell::exact(self.n, out.radii.len())
    }
    /// Run `kernel` on the first lane after `preflight`, recording the attempt.
    fn single(
        &self,
        workspace: &mut LinearWorkspace<S>,
        scope: SolveScope<'_, '_>,
        preflight: impl FnOnce() -> Result<()>,
        kernel: impl FnOnce(
            &mut SweepScratch<'_, S::Real>,
            &mut LaneView<'_, S>,
            &mut SolveContext<'_, '_>,
        ) -> Result<LinearVerificationReport<S::Real>>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let shape = self.shape();
        workspace.core.run(
            scope,
            self.identity(),
            &self.options,
            || {
                self.validate()?;
                preflight()?;
                self.workspace_requirements(1)
            },
            |_, _| Ok(()),
            |lanes, sweep, _, _, context| {
                kernel(&mut sweep.view(shape), &mut lanes[0].view(shape), context)
            },
            LinearVerificationReport::is_verified,
        )
    }
    /// Proves that `A` is nonsingular and bounds `‖A‖`, `‖R‖`, `‖I − RA‖`,
    /// `‖A⁻¹‖` and the condition number, without solving.
    ///
    /// `Verified` means that `q = ‖I − RA‖ < 1`. Each column of the matrix
    /// counts as one iteration and three operator applications.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operators changed shape or returned an invalid
    ///   enclosure, or if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - Errors of the operators, and `AllocationFailure` for the workspace.
    pub fn bounds(&self) -> Result<LinearVerificationReport<S::Real>> {
        self.bounds_scoped(
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::bounds`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::bounds`], and those of
    ///   [`LinearWorkspace::try_reserve`] if the workspace must grow.
    pub fn bounds_scoped<'s, 'c>(
        &self,
        workspace: &mut LinearWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        self.single(
            workspace,
            scope.into(),
            || Ok(()),
            |sweep, _, context| self.sweep(None, true, sweep, context),
        )
    }
    /// Checks a proposed solution and error bound for the right-hand side `rhs`.
    ///
    /// The status is `Verified` if `q < 1` and the radii pass the test
    /// described on [`LinearVerifier`], `Rejected` if `q < 1` but they do
    /// not, and `Inconclusive` otherwise. A rejection does not mean that `A` is
    /// singular. For order `n > 0` this costs `2n + 2` operator applications and
    /// `n` iterations, plus any work inside the operators. The report contains
    /// `‖A‖` and `q`; use [`Self::bounds`] for the other norms.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or the proposal does not have length `n`.
    /// - `NonFinite` if a center or radius is NaN or infinite.
    /// - `InvalidInput` if a radius is negative, a real system has a nonzero
    ///   imaginary radius, `rhs` is unbounded or malformed, the operators
    ///   changed shape or returned an invalid enclosure, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the operators.
    pub fn verify(
        &self,
        rhs: &[IntervalBox<S::Real>],
        certificate: LinearCertificateView<'_, S>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        self.verify_scoped(
            rhs,
            certificate,
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify`], and those of
    ///   [`LinearWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_scoped<'s, 'c>(
        &self,
        rhs: &[IntervalBox<S::Real>],
        certificate: LinearCertificateView<'_, S>,
        workspace: &mut LinearWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        self.single(
            workspace,
            scope.into(),
            || {
                self.rhs(rhs)?;
                certificate.validate(self.n, scioxide_core::argument!("certificate"))
            },
            |sweep, lane, context| {
                let report = self.sweep(Some(certificate.radii), false, sweep, context)?;
                let report =
                    self.residual(rhs, certificate.center, report, &mut lane.proof, context)?;
                let report = check_enclosure::<S>(
                    report,
                    certificate.center,
                    certificate.radii,
                    lane.proof.z,
                    sweep.image,
                    sweep.e_rows,
                )?;
                commit(report, context)
            },
        )
    }
    /// Computes error radii around your approximate solution `center` and
    /// writes the certified solution to `out`.
    ///
    /// For order `n > 0` this costs `2n + 2` operator applications and `n`
    /// iterations, plus any work inside the operators. `out` changes only if
    /// the status is `Verified`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs`, `center` or `out` does not have length
    ///   `n`.
    /// - `NonFinite` if `center` contains NaN or infinity.
    /// - `InvalidInput` if `rhs` is unbounded or malformed, the operators
    ///   changed shape or returned an invalid enclosure, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the operators, and those of
    ///   [`LinearWorkspace::try_reserve`] if the workspace must grow.
    pub fn propose_into<'s, 'c>(
        &self,
        rhs: &[IntervalBox<S::Real>],
        center: &[S],
        out: LinearCertificateViewMut<'_, S>,
        workspace: &mut LinearWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let report = self.single(
            workspace,
            scope.into(),
            || {
                self.rhs(rhs)?;
                self.output(&out)?;
                crate::shell::exact(self.n, center.len())?;
                if center.iter().any(|x| !x.has_finite_components()) {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Input,
                        "initial approximation must be finite",
                    ));
                }
                Ok(())
            },
            |sweep, lane, context| {
                let report = self.sweep(None, false, sweep, context)?;
                lane.center.copy_from_slice(center);
                let regularity = Regularity {
                    report,
                    rows: sweep.e_rows,
                };
                let report = self.proposal(
                    rhs,
                    center,
                    lane.radii,
                    &mut lane.proof,
                    regularity,
                    context,
                )?;
                commit(report, context)
            },
        )?;
        self.publish(report, &workspace.core.lanes[0], out);
        Ok(report)
    }
    /// Bounds the error of `center` for the right-hand side `rhs`; see
    /// [`Self::propose_into`].
    ///
    /// Allocates the workspace and the certificate and runs sequentially. The
    /// certificate is present exactly when the status is `Verified`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::propose_into`].
    /// - `AllocationFailure` if the storage cannot be reserved.
    #[expect(
        clippy::type_complexity,
        reason = "the report and the conditionally published certificate are one owned result"
    )]
    pub fn propose(
        &self,
        rhs: &[IntervalBox<S::Real>],
        center: &[S],
    ) -> Result<(
        LinearVerificationReport<S::Real>,
        Option<LinearCertificate<S>>,
    )> {
        let mut certificate = LinearCertificate::new(self.n)?;
        let report = self.propose_into(
            rhs,
            center,
            certificate.view_mut(),
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, report.is_verified().then_some(certificate)))
    }
    /// Computes and certifies the solution for the right-hand side `rhs`; see
    /// [`Self::solve_into`].
    ///
    /// Allocates the workspace and the certificate and runs sequentially. The
    /// certificate is present exactly when the status is `Verified`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`].
    /// - `AllocationFailure` if the storage cannot be reserved.
    #[expect(
        clippy::type_complexity,
        reason = "the report and the conditionally published certificate are one owned result"
    )]
    pub fn solve(
        &self,
        rhs: &[IntervalBox<S::Real>],
    ) -> Result<(
        LinearVerificationReport<S::Real>,
        Option<LinearCertificate<S>>,
    )> {
        let mut certificate = LinearCertificate::new(self.n)?;
        let report = self.solve_into(
            rhs,
            certificate.view_mut(),
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, report.is_verified().then_some(certificate)))
    }
    /// Computes and certifies the solution for the right-hand side `rhs` and
    /// writes it to `out`.
    ///
    /// The starting point is `x₀ = R·mid(b)`, the point preconditioner applied
    /// to the midpoint of `rhs`. One correction step `x̃ = x₀ + z`, with
    /// `z ∋ R(b − Ax₀)`, refines it, and the error radii around `x̃` follow from
    /// the contraction bound described on [`LinearVerifier`]. For order
    /// `n > 0` this costs `2n + 3` operator applications and `n` iterations,
    /// plus any work inside the operators. `out` changes only if the status is
    /// `Verified`. The residual bound in the report is that of `x₀` updated by
    /// the change of center, and can be larger than the residual of `x̃`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `out` does not have length `n`.
    /// - `InvalidInput` if `rhs` is unbounded or malformed, the operators
    ///   changed shape or returned an invalid enclosure, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the operators, and those of
    ///   [`LinearWorkspace::try_reserve`] if the workspace must grow.
    ///
    /// A failure of the point application of `R` gives an `Inconclusive`
    /// status with [`LinearProofFailure::SeedFailure`], not an error.
    pub fn solve_into<'s, 'c>(
        &self,
        rhs: &[IntervalBox<S::Real>],
        out: LinearCertificateViewMut<'_, S>,
        workspace: &mut LinearWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let report = self.single(
            workspace,
            scope.into(),
            || {
                self.rhs(rhs)?;
                self.output(&out)
            },
            |sweep, lane, context| {
                let report = self.sweep(None, false, sweep, context)?;
                self.solve_kernel(
                    rhs,
                    lane,
                    Regularity {
                        report,
                        rows: sweep.e_rows,
                    },
                    context,
                )
            },
        )?;
        self.publish(report, &workspace.core.lanes[0], out);
        Ok(report)
    }
    /// Computes and certifies the solutions for the `k` columns of `rhs`,
    /// writing one report per column.
    ///
    /// The matrix bounds are computed once. The columns then run as independent
    /// proofs, in parallel under the options' execution policy; the remaining
    /// budgets are divided by column index, so the reports do not depend on the
    /// policy or the number of workers. For order `n > 0` this costs `2n + 3k`
    /// operator applications and `n` iterations, plus any work inside the
    /// operators. `centers` and `radii` change only if every column is
    /// `Verified`; `reports` is written whenever the call returns `Ok`.
    ///
    /// ```
    /// use scioxide_core::{BlockVectorView, BlockVectorViewMut, Complex, ExecutionContext};
    /// use scioxide_linalg::MatrixView;
    /// use scioxide_validated::{
    ///     DenseIntervalOperator, LinearVerificationReport, ValidatedOptions, LinearVerifier,
    ///     point_enclosure,
    /// };
    ///
    /// // 2x = 1 and 2x = 4.
    /// let (a, r) = ([2.0_f64], [0.5_f64]);
    /// let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a)?)?;
    /// let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r)?)?;
    /// let verifier = LinearVerifier::new(&system, &inverse, ValidatedOptions::new())?;
    /// let rhs = [point_enclosure(1.0)?, point_enclosure(4.0)?];
    /// let (mut centers, mut radii) = ([0.0; 2], [Complex::new(0.0, 0.0); 2]);
    /// let mut workspace = verifier.workspace(2)?;
    /// let mut reports = [LinearVerificationReport::not_run(); 2];
    /// verifier.solve_block_into(
    ///     BlockVectorView::new(&rhs, 1, 2)?,
    ///     &mut BlockVectorViewMut::new(&mut centers, 1, 2)?,
    ///     &mut BlockVectorViewMut::new(&mut radii, 1, 2)?,
    ///     &mut reports,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!(reports.iter().all(|report| report.is_verified()));
    /// assert_eq!(centers, [0.5, 2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the blocks do not have `n` rows, or `centers`,
    ///   `radii` and `reports` do not have `k` columns.
    /// - `InvalidInput` if a column of `rhs` is unbounded or malformed, the
    ///   operators changed shape or returned an invalid enclosure, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of the operators and the executor, and those of
    ///   [`LinearWorkspace::try_reserve`] if the workspace must grow.
    pub fn solve_block_into<'s, 'c>(
        &self,
        rhs: BlockVectorView<'_, IntervalBox<S::Real>>,
        centers: &mut BlockVectorViewMut<'_, S>,
        radii: &mut BlockVectorViewMut<'_, Complex<S::Real>>,
        reports: &mut [LinearVerificationReport<S::Real>],
        workspace: &mut LinearWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<()> {
        let count = rhs.right_hand_sides();
        let report_count = reports.len();
        let shape = self.shape();
        let verified = workspace.core.verify_batch(
            scope.into(),
            self.identity(),
            &self.options,
            || {
                self.validate()?;
                crate::shell::exact(self.n, rhs.rows())?;
                crate::shell::exact(self.n, centers.rows())?;
                crate::shell::exact(count, centers.right_hand_sides())?;
                crate::shell::exact(self.n, radii.rows())?;
                crate::shell::exact(count, radii.right_hand_sides())?;
                crate::shell::exact(count, report_count)?;
                validate_boxes::<S>(rhs.as_slice(), true, scioxide_core::argument!("rhs"))?;
                self.workspace_requirements(count)
            },
            reports,
            |sweep, context| {
                let report = self.sweep(None, false, &mut sweep.view(shape), context)?;
                Ok(if report.is_verified() {
                    crate::shell::Prepared::Ready(report)
                } else {
                    crate::shell::Prepared::Inconclusive(report)
                })
            },
            &|index, lane, sweep, report, context| {
                self.solve_kernel(
                    rhs.column(index)?,
                    &mut lane.view(shape),
                    Regularity {
                        report: *report,
                        rows: sweep.defect_rows(self.n),
                    },
                    context,
                )
            },
        )?;
        if verified {
            for (index, lane) in workspace.core.lanes[..count].iter().enumerate() {
                let (center, radius) = lane.staged(self.n);
                centers.column_mut(index)?.copy_from_slice(center);
                radii.column_mut(index)?.copy_from_slice(radius);
            }
        }
        Ok(())
    }
    /// Copy a verified staged certificate into `out`.
    fn publish(
        &self,
        report: LinearVerificationReport<S::Real>,
        lane: &super::workspace::Lane<S>,
        out: LinearCertificateViewMut<'_, S>,
    ) {
        if report.is_verified() {
            let (center, radii) = lane.staged(self.n);
            out.center.copy_from_slice(center);
            out.radii.copy_from_slice(radii);
        }
    }
    /// Seed, refine and certify one right-hand side into the lane's staged certificate.
    fn solve_kernel(
        &self,
        rhs: &[IntervalBox<S::Real>],
        w: &mut LaneView<'_, S>,
        regularity: Regularity<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        if !regularity.report.is_verified() {
            return Ok(regularity.report);
        }
        if let Some(stop) = stop(context)? {
            return Ok(LinearVerificationReport::without_diagnostics(stop));
        }
        for (slot, &b) in w.midpoint.iter_mut().zip(rhs) {
            *slot = midpoint(b, scioxide_core::argument!("rhs"))?;
        }
        if self.n != 0 {
            if !context
                .record_provider(EvaluationRole::LinearSolve)?
                .is_granted()
            {
                return Ok(LinearVerificationReport::without_diagnostics(
                    LinearVerificationStatus::Inconclusive(LinearInconclusive::EvaluationLimit),
                ));
            }
            w.center.fill(S::from_real(S::Real::NAN));
            let invocation = self.inverse.apply_into_scoped(
                w.midpoint,
                w.center,
                w.scalar,
                &mut OperatorContext::new(context),
            );
            self.validate()?;
            let inconclusive = |status| LinearVerificationReport {
                status,
                ..regularity.report
            };
            if let Some(status) = stop(context)? {
                return Ok(inconclusive(status));
            }
            let seed_failure = || {
                inconclusive(LinearVerificationStatus::Inconclusive(
                    LinearInconclusive::Reason(LinearProofFailure::SeedFailure),
                ))
            };
            match invocation {
                Ok(Invocation::Evaluated(())) => {}
                Ok(Invocation::BudgetExhausted(_)) => {
                    return Ok(inconclusive(LinearVerificationStatus::Inconclusive(
                        LinearInconclusive::EvaluationLimit,
                    )));
                }
                Ok(Invocation::ExecutionStopped(reason)) => {
                    return Ok(inconclusive(LinearVerificationStatus::Inconclusive(
                        LinearInconclusive::Stopped(reason),
                    )));
                }
                Err(error) => {
                    if let Some(reason) = crate::shell::stopped(&error) {
                        return Ok(inconclusive(LinearVerificationStatus::Inconclusive(
                            LinearInconclusive::Stopped(reason),
                        )));
                    }
                    if matches!(
                        error.kind(),
                        scioxide_core::SciOxideErrorKind::NonFinite
                            | scioxide_core::SciOxideErrorKind::NumericalInstability
                            | scioxide_core::SciOxideErrorKind::ConvergenceFailure
                            | scioxide_core::SciOxideErrorKind::Singular
                    ) {
                        return Ok(seed_failure());
                    }
                    return Err(error);
                }
            }
            if w.center.iter().any(|x| !x.has_finite_components()) {
                return Ok(seed_failure());
            }
        }
        // Enclose the original-system residual once at the ordinary seed. For
        // an exact center displacement d, z(x+d) = z(x) - d + (I-RA)d.
        // The independently computed defect rows therefore transport this proof
        // through an untrusted center correction without another application.
        let mut report = self.residual(rhs, w.center, regularity.report, &mut w.proof, context)?;
        if !report.is_verified() {
            return Ok(report);
        }
        let matrix = report
            .bounds
            .ok_or_else(|| crate::shell::breakdown("missing completed bounds"))?
            .matrix;
        for ((next, &center), &z) in w
            .midpoint
            .iter_mut()
            .zip(w.center.iter())
            .zip(w.proof.z.iter())
        {
            *next = center + midpoint::<S>(z, scioxide_core::argument!("inverse"))?;
            if !next.has_finite_components() {
                report.status =
                    LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
                return Ok(report);
            }
        }
        if w.midpoint != w.center {
            let displacement = w.midpoint.iter().zip(w.center.iter()).try_fold(
                S::Real::ZERO,
                |maximum, (&next, &center)| {
                    let (lo, hi) =
                        displacement_component(next.real_component(), center.real_component())?;
                    let maximum = maximum.max(lo.abs()).max(hi.abs());
                    Ok::<_, SciOxideError>(if S::IS_REAL {
                        maximum
                    } else {
                        let (lo, hi) = displacement_component(
                            next.imaginary_component(),
                            center.imaginary_component(),
                        )?;
                        maximum.max(lo.abs()).max(hi.abs())
                    })
                },
            )?;
            if !displacement.is_finite() {
                report.status =
                    LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
                return Ok(report);
            }
            for (((z, &next), &center), &row) in w
                .proof
                .z
                .iter_mut()
                .zip(w.midpoint.iter())
                .zip(w.center.iter())
                .zip(regularity.rows)
            {
                let tail = mul_up(row, displacement);
                if !tail.is_finite() {
                    report.status =
                        LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
                    return Ok(report);
                }
                let prior = unpack(*z);
                *z = pack(Box2 {
                    re: transport_component(
                        prior.re,
                        next.real_component(),
                        center.real_component(),
                        tail,
                    )?,
                    im: if S::IS_REAL {
                        prior.im
                    } else {
                        transport_component(
                            prior.im,
                            next.imaginary_component(),
                            center.imaginary_component(),
                            tail,
                        )?
                    },
                });
                if !unpack(*z).finite() {
                    report.status =
                        LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
                    return Ok(report);
                }
            }
            // ||r(x+d)|| <= ||r(x)|| + ||A|| ||d||. This diagnostic remains
            // rigorous even though it need not shrink with the refined center.
            report.residual = Some(add_up(
                report
                    .residual
                    .ok_or_else(|| crate::shell::breakdown("missing residual bound"))?,
                mul_up(matrix, displacement),
            ));
            w.center.copy_from_slice(w.midpoint);
        }
        report.correction = Some(box_norm::<S>(w.proof.z)?);
        let report = self.enclose(w.center, w.radii, w.proof.z, regularity, report)?;
        commit(report, context)
    }
    /// Generate and verify radii.
    fn proposal(
        &self,
        rhs: &[IntervalBox<S::Real>],
        center: &[S],
        radii: &mut [Complex<S::Real>],
        scratch: &mut ProofScratch<'_, S::Real>,
        regularity: Regularity<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let first = self.residual(rhs, center, regularity.report, scratch, context)?;
        if !first.is_verified() {
            return Ok(first);
        }
        self.enclose(center, radii, scratch.z, regularity, first)
    }
    /// Generate the contraction-bound radii around `center` from the in-call
    /// regularity proof and the independently enclosed correction `z`.
    fn enclose(
        &self,
        center: &[S],
        radii: &mut [Complex<S::Real>],
        z: &[IntervalBox<S::Real>],
        regularity: Regularity<'_, S::Real>,
        first: LinearVerificationReport<S::Real>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let unbounded = LinearVerificationReport {
            status: LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded),
            ..first
        };
        let rho = contraction_radius(&first)?;
        if !rho.is_finite() {
            return Ok(unbounded);
        }
        // Epsilon-inflation leaves room for `verify`, whose correction at the
        // published center may differ from the transported one by a few units
        // in the last place; any widening of a proved box remains proved.
        let inflation = S::Real::ONE + S::Real::EPSILON * (S::Real::TWO + S::Real::TWO);
        for (((r, &z), &row), &center) in radii.iter_mut().zip(z).zip(regularity.rows).zip(center) {
            let bound = contraction_component::<S>(z, row, rho)?;
            *r = Complex::new(mul_up(bound.re, inflation), mul_up(bound.im, inflation));
            if !r.re.is_finite() || !r.im.is_finite() || !finite_enclosure(center, r)? {
                return Ok(unbounded);
            }
        }
        Ok(first)
    }
    /// Recompute the original residual and correction using completed regularity.
    fn residual(
        &self,
        rhs: &[IntervalBox<S::Real>],
        center: &[S],
        mut report: LinearVerificationReport<S::Real>,
        s: &mut ProofScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        if !report.is_verified() {
            return Ok(report);
        }
        if self.n == 0 {
            report.residual = Some(S::Real::ZERO);
            report.correction = Some(S::Real::ZERO);
            return Ok(report);
        }
        for (slot, &x) in s.input.iter_mut().zip(center) {
            *slot = pack(Box2::point(x)?.neg()?);
        }
        let reason = application(
            self.system,
            s.input,
            Some(rhs),
            s.residual,
            s.provider,
            context,
            EvaluationRole::Residual,
        )?;
        self.validate()?;
        if let Some(reason) = reason {
            report.status = reason;
            return Ok(report);
        }
        report.residual = Some(box_norm::<S>(s.residual)?);
        if !s.residual.iter().all(|&x| unpack(x).finite()) {
            report.status = LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
            return Ok(report);
        }
        let reason = application(
            self.inverse,
            s.residual,
            None,
            s.z,
            s.provider,
            context,
            EvaluationRole::Preconditioner,
        )?;
        self.validate()?;
        if let Some(reason) = reason {
            report.status = reason;
            return Ok(report);
        }
        report.correction = Some(box_norm::<S>(s.z)?);
        if !s.z.iter().all(|&x| unpack(x).finite()) {
            report.status = LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
        }
        Ok(report)
    }

    /// One streamed matrix proof. Weighted images of external radii are formed
    /// during this sweep; generated radii use the stored row bounds instead.
    fn sweep(
        &self,
        radii: Option<&[Complex<S::Real>]>,
        norms: bool,
        s: &mut SweepScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<LinearVerificationReport<S::Real>> {
        let mut report =
            LinearVerificationReport::without_diagnostics(LinearVerificationStatus::Verified);
        if let Some(reason) = stop(context)? {
            report.status = reason;
            return Ok(report);
        }
        let zero_value = zero()?;
        s.a_rows.fill(S::Real::ZERO);
        s.r_rows.fill(S::Real::ZERO);
        s.e_rows.fill(S::Real::ZERO);
        s.image.fill(Complex::new(S::Real::ZERO, S::Real::ZERO));
        macro_rules! apply {
            ($operator:expr,$input:expr,$bias:expr,$output:expr,$role:expr) => {
                let reason = application(
                    $operator,
                    $input,
                    $bias,
                    $output,
                    &mut s.provider,
                    context,
                    $role,
                )?;
                self.validate()?;
                if let Some(reason) = reason {
                    report.status = reason;
                    return Ok(report);
                }
            };
        }
        for j in 0..self.n {
            if let Some(reason) = stop(context)? {
                report.status = reason;
                return Ok(report);
            }
            if !context.record_iteration()?.is_granted() {
                report.status =
                    LinearVerificationStatus::Inconclusive(LinearInconclusive::IterationLimit);
                return Ok(report);
            }
            s.input.fill(zero_value);
            s.input[j] = point_enclosure(S::ONE)?;
            apply!(
                self.system,
                &s.input,
                None,
                &mut s.a,
                EvaluationRole::LinearOperator
            );
            if !s.a.iter().all(|&x| unpack(x).finite()) {
                report.status =
                    LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
                return Ok(report);
            }
            apply!(
                self.inverse,
                &s.a,
                None,
                &mut s.r,
                EvaluationRole::Preconditioner
            );
            for i in 0..self.n {
                let a = unpack(s.a[i]).radii()?;
                s.a_rows[i] = add_up(s.a_rows[i], add_up(a.0, a.1));
                let e = defect_entry(s.input[i], s.r[i], &mut s.e_rows[i])?;
                let (re, im) = e.radii()?;
                if let Some(radii) = radii {
                    let r = radii[j];
                    s.image[i].re =
                        add_up(s.image[i].re, add_up(mul_up(re, r.re), mul_up(im, r.im)));
                    s.image[i].im =
                        add_up(s.image[i].im, add_up(mul_up(im, r.re), mul_up(re, r.im)));
                }
            }
            if norms {
                apply!(
                    self.inverse,
                    &s.input,
                    None,
                    &mut s.r,
                    EvaluationRole::Preconditioner
                );
                for (i, &r) in s.r.iter().enumerate() {
                    let (re, im) = unpack(r).radii()?;
                    s.r_rows[i] = add_up(s.r_rows[i], add_up(re, im));
                }
            }
        }
        let maximum = |v: &[S::Real]| v.iter().copied().fold(S::Real::ZERO, RealScalar::max);
        let a = maximum(s.a_rows);
        let r = norms.then(|| maximum(s.r_rows));
        let q = maximum(s.e_rows);
        let inverse = if let Some(r) = r.filter(|r| r.is_finite())
            && q < S::Real::ONE
        {
            let denom = D::point(S::Real::ONE)?.sub_admitted(&D::point(q)?)?;
            let v = upper(D::point(r)?.div_admitted(&denom)?);
            v.is_finite().then_some(v)
        } else {
            None
        };
        let condition = match inverse {
            Some(v) if a.is_finite() => {
                let c = mul_up(a, v);
                c.is_finite().then_some(c)
            }
            _ => None,
        };
        report.bounds = Some(LinearBounds {
            matrix: a,
            preconditioner: r,
            defect: q,
            inverse,
            condition,
        });
        if !q.is_finite() {
            report.status = LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
        } else if q >= S::Real::ONE {
            report.status = LinearVerificationStatus::Inconclusive(LinearInconclusive::Reason(
                LinearProofFailure::RegularityNotProved,
            ));
        }
        if let Some(reason) = stop(context)? {
            report.status = reason;
        }
        Ok(report)
    }
}

/// Outward endpoints of an exact center displacement. The centers are finite;
/// an overflowing displacement becomes an unbounded proof outcome.
#[inline]
fn displacement_component<R: RealScalar>(next: R, prior: R) -> Result<(R, R)> {
    Ok((-add_up(-next, prior), add_up(next, -prior)))
}

/// Enclose z - d + [-tail, tail] directly from directed endpoints. All arguments
/// were checked in this call; no decoration or provider information is lost.
#[inline]
fn transport_component<R: RealScalar>(z: D<R>, next: R, prior: R, tail: R) -> Result<D<R>> {
    let (lo, hi) = bounded(z, scioxide_core::argument!("self"))?;
    let (dl, dh) = displacement_component(next, prior)?;
    D::result(
        -add_up(add_up(-lo, dh), tail),
        add_up(add_up(hi, -dl), tail),
        crate::Decoration::Com,
    )
}

/// Both endpoints of center +/- radius are finite exactly when the upward
/// bound |center| + radius is finite, independently in each component.
fn finite_enclosure<S: NumericScalar>(center: S, radius: &Complex<S::Real>) -> Result<bool> {
    Ok(add_up(center.real_component().abs(), radius.re).is_finite()
        && (S::IS_REAL || add_up(center.imaginary_component().abs(), radius.im).is_finite()))
}

/// Upper bound `rho >= ||z|| / (1-q)` on the error norm of a report whose
/// completed sweep proved `q < 1`; it is infinite when no finite bound exists.
///
/// The sweep bounds row i of `|I-RA|` by `row_i <= q` in the real-component norm
/// for every A of the system and R of the preconditioner, so RA, and hence A, is
/// nonsingular. The error `e = x* - x` of every solution then satisfies
/// `e = R(b-Ax) + (I-RA)e` with `R(b-Ax)` in z. So `||e|| <= ||z|| + q ||e||`,
/// giving `||e|| <= rho`, and each real or imaginary part obeys
/// `|e_i| <= |z_i| + row_i ||e|| <= |z_i| + row_i rho`. Directed rounding keeps
/// every step an upper bound, so this bound needs no invariance check.
fn contraction_radius<R: RealScalar>(report: &LinearVerificationReport<R>) -> Result<R> {
    let defect = report
        .bounds
        .ok_or_else(|| crate::shell::breakdown("missing completed bounds"))?
        .defect;
    let correction = report
        .correction
        .ok_or_else(|| crate::shell::breakdown("missing correction bound"))?;
    let denominator = D::point(R::ONE)?.sub_admitted(&D::point(defect)?)?;
    Ok(upper(D::point(correction)?.div_admitted(&denominator)?))
}

/// Component radii `|z_i| + row_i rho` of the contraction bound for a finite `rho`.
fn contraction_component<S: NumericScalar>(
    z: IntervalBox<S::Real>,
    row: S::Real,
    rho: S::Real,
) -> Result<Complex<S::Real>> {
    let (re, im) = unpack(z).radii()?;
    let tail = mul_up(row, rho);
    Ok(Complex::new(
        add_up(re, tail),
        if S::IS_REAL {
            S::Real::ZERO
        } else {
            add_up(im, tail)
        },
    ))
}

/// Pure external-certificate check against its freshly streamed weighted defect
/// image and row bounds. The box is proved when it is invariant,
/// `|z_i| + (|I-RA| r)_i <= r_i` for every i, or contains the contraction bound
/// of [`contraction_radius`]. This invokes no providers and trusts no prior call.
fn check_enclosure<S: NumericScalar>(
    mut report: LinearVerificationReport<S::Real>,
    center: &[S],
    radii: &[Complex<S::Real>],
    z: &[IntervalBox<S::Real>],
    image: &[Complex<S::Real>],
    rows: &[S::Real],
) -> Result<LinearVerificationReport<S::Real>> {
    if !report.is_verified() {
        return Ok(report);
    }
    let rho = contraction_radius(&report)?;
    let (mut invariant, mut contained) = (true, rho.is_finite());
    for ((((&center, radius), &z), image), &row) in
        center.iter().zip(radii).zip(z).zip(image).zip(rows)
    {
        if !radius.re.is_finite() || !radius.im.is_finite() || !finite_enclosure(center, radius)? {
            report.status = LinearVerificationStatus::Inconclusive(LinearInconclusive::Unbounded);
            break;
        }
        let (zr, zi) = unpack(z).radii()?;
        invariant =
            invariant && add_up(zr, image.re) <= radius.re && add_up(zi, image.im) <= radius.im;
        if contained {
            let bound = contraction_component::<S>(z, row, rho)?;
            contained = bound.re <= radius.re && bound.im <= radius.im;
        }
        if !invariant && !contained {
            report.status = LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow);
            break;
        }
    }
    Ok(report)
}
fn box_norm<S: NumericScalar>(values: &[IntervalBox<S::Real>]) -> Result<S::Real> {
    values.iter().try_fold(S::Real::ZERO, |norm, &x| {
        let x = unpack(x);
        let re = crate::rectangle::magnitude(x.re);
        Ok(if S::IS_REAL {
            norm.max(re)
        } else {
            norm.max(re).max(crate::rectangle::magnitude(x.im))
        })
    })
}
/// Pending stop.
fn stop(context: &SolveContext<'_, '_>) -> Result<Option<LinearVerificationStatus>> {
    Ok(context
        .stop_reason()?
        .map(|reason| LinearVerificationStatus::Inconclusive(LinearInconclusive::Stopped(reason))))
}
/// Apply one enclosing provider of the verifier (`self`).
fn application<S: NumericScalar>(
    operator: &(dyn IntervalLinearOperator<S> + Sync),
    input: &[IntervalBox<S::Real>],
    addend: Option<&[IntervalBox<S::Real>]>,
    output: &mut [IntervalBox<S::Real>],
    scratch: &mut [IntervalBox<S::Real>],
    context: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
) -> Result<Option<LinearVerificationStatus>> {
    if let Some(reason) = stop(context)? {
        return Ok(Some(reason));
    }
    if !context.record_provider(role)?.is_granted() {
        return Ok(Some(LinearVerificationStatus::Inconclusive(
            LinearInconclusive::EvaluationLimit,
        )));
    }
    output.fill(sentinel());
    let needed = operator.enclosure_workspace_len()?;
    if needed > scratch.len() {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "provider changed workspace requirement",
        ));
    }
    match operator.apply_enclosure_into_scoped(
        input,
        addend,
        output,
        &mut scratch[..needed],
        &mut scioxide_core::OperatorContext::new(context),
    ) {
        Ok(scioxide_core::Invocation::Evaluated(())) => {}
        Ok(scioxide_core::Invocation::BudgetExhausted(_)) => {
            return Ok(Some(LinearVerificationStatus::Inconclusive(
                LinearInconclusive::EvaluationLimit,
            )));
        }
        Ok(scioxide_core::Invocation::ExecutionStopped(reason)) => {
            return Ok(Some(LinearVerificationStatus::Inconclusive(
                LinearInconclusive::Stopped(reason),
            )));
        }
        Err(e) => {
            return match crate::shell::stopped(&e) {
                Some(reason) => Ok(Some(LinearVerificationStatus::Inconclusive(
                    LinearInconclusive::Stopped(reason),
                ))),
                None => Err(e),
            };
        }
    }
    crate::environment::check()?;
    validate_boxes::<S>(output, false, scioxide_core::argument!("self"))?;
    stop(context)
}

/// Commit a verified certificate's trust in the epoch.
fn commit<R: RealScalar>(
    mut report: LinearVerificationReport<R>,
    context: &mut SolveContext<'_, '_>,
) -> Result<LinearVerificationReport<R>> {
    if let Some(reason) = stop(context)? {
        report.status = reason;
    }
    if report.is_verified() {
        context.commit_accepted(PartialResultTrust::CertificateVerified)?;
    }
    Ok(report)
}
