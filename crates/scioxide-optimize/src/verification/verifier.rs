//! Prepared independent verification of LP and QP claims.
use super::kernel::{interrupted, lp_session, qp_session, reported_claim, validate_claim};
use super::{
    LpVerificationProblem, ProgrammingCertificate, QpVerificationProblem, VerificationReport,
    VerificationStatus, VerificationWorkspace,
};
use crate::context::SolveSession;
use crate::{OptimalityTolerances, OptimizeTermination, ProgrammingResult};

use scioxide_core::{
    AttemptKind, Budget, BudgetKind, ExecutionContext, NumericScalar, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveScope,
};

/// Sets the tolerances and the evaluation limit of [`ProgrammingVerifier`].
///
/// The default tolerances are [`OptimalityTolerances::default`]: `√ε` for every
/// role, with no floor for rounding error. Each application of a constraint or
/// Hessian operator counts as one evaluation. Without a limit, only the limits
/// of a parent solve context apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerificationOptions<S: RealScalar> {
    tolerances: OptimalityTolerances<S>,
    max_evaluations: Option<Budget>,
}

impl<S: RealScalar> Default for VerificationOptions<S> {
    fn default() -> Self {
        Self {
            tolerances: OptimalityTolerances::default(),
            max_evaluations: None,
        }
    }
}

impl<S: RealScalar> VerificationOptions<S> {
    /// Sets the tolerances of the checks.
    ///
    /// Constraint and bound residuals use [`OptimalityTolerances::feasibility`],
    /// multiplier signs [`OptimalityTolerances::dual_feasibility`], the
    /// Lagrangian gradient and the Farkas combination
    /// [`OptimalityTolerances::gradient`], multiplier-slack products
    /// [`OptimalityTolerances::complementarity`], objective gaps and certificate
    /// margins [`OptimalityTolerances::gap`], and the convexity of the Hessian
    /// [`OptimalityTolerances::rank`].
    #[must_use]
    pub const fn with_tolerances(mut self, tolerances: OptimalityTolerances<S>) -> Self {
        self.tolerances = tolerances;
        self
    }
    /// Sets the maximum number of operator applications in one check.
    #[must_use]
    pub const fn with_max_evaluations(mut self, limit: Budget) -> Self {
        self.max_evaluations = Some(limit);
        self
    }
    /// Returns the tolerances of the checks.
    pub const fn tolerances(&self) -> OptimalityTolerances<S> {
        self.tolerances
    }
    /// Returns the maximum number of operator applications, if set.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }
}

mod sealed {
    use super::{
        NumericScalar, OptimalityTolerances, ProgrammingCertificate, RealScalar, SciOxideError,
        SolveContext, VerificationReport, VerificationWorkspace,
    };

    pub trait Sealed<S: RealScalar + NumericScalar<Real = S>> {
        fn dimension(&self) -> usize;
        fn prepare(&self) -> Result<VerificationWorkspace<S>, SciOxideError>;
        fn validate(
            &self,
            certificate: ProgrammingCertificate<'_, S>,
            workspace: &VerificationWorkspace<S>,
        ) -> Result<(), SciOxideError>;
        fn check(
            &self,
            certificate: ProgrammingCertificate<'_, S>,
            tolerances: OptimalityTolerances<S>,
            workspace: &mut VerificationWorkspace<S>,
            context: &mut SolveContext<'_, '_>,
        ) -> Result<VerificationReport<S>, SciOxideError>;
    }
}

/// Marks the problem types that [`ProgrammingVerifier`] accepts.
///
/// The trait is sealed; [`crate::LpProblem`], [`crate::QpProblem`],
/// [`LpVerificationProblem`] and [`QpVerificationProblem`] implement it.
pub trait VerificationProblem<S: RealScalar + NumericScalar<Real = S>>: sealed::Sealed<S> {}

impl<S: RealScalar + NumericScalar<Real = S>> sealed::Sealed<S> for LpVerificationProblem<'_, S> {
    fn dimension(&self) -> usize {
        LpVerificationProblem::dimension(self)
    }
    fn prepare(&self) -> Result<VerificationWorkspace<S>, SciOxideError> {
        VerificationWorkspace::for_lp(self)
    }
    fn validate(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        workspace: &VerificationWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        validate_claim(self, None, certificate, workspace)
    }
    fn check(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        tolerances: OptimalityTolerances<S>,
        workspace: &mut VerificationWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<VerificationReport<S>, SciOxideError> {
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        interrupted(
            lp_session(self, certificate, tolerances, workspace, Some(&session)),
            &session,
        )
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> VerificationProblem<S>
    for LpVerificationProblem<'_, S>
{
}

impl<S: RealScalar + NumericScalar<Real = S>> sealed::Sealed<S> for QpVerificationProblem<'_, S> {
    fn dimension(&self) -> usize {
        self.linear().dimension()
    }
    fn prepare(&self) -> Result<VerificationWorkspace<S>, SciOxideError> {
        VerificationWorkspace::for_qp(self)
    }
    fn validate(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        workspace: &VerificationWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        validate_claim(self.linear(), Some(self), certificate, workspace)
    }
    fn check(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        tolerances: OptimalityTolerances<S>,
        workspace: &mut VerificationWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<VerificationReport<S>, SciOxideError> {
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        interrupted(
            qp_session(self, certificate, tolerances, workspace, Some(&session)),
            &session,
        )
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> VerificationProblem<S>
    for QpVerificationProblem<'_, S>
{
}

/// Describes what [`ProgrammingVerifier`] checks: a certificate, or a solver result.
///
/// Convert from a [`ProgrammingCertificate`] to check the certificate alone.
/// Convert from a [`ProgrammingResult`] to check its certificate, and also that
/// its termination matches the kind of certificate and that its reported
/// objective agrees with the recomputed one within the gap tolerance. A result
/// without a certificate is rejected.
#[derive(Debug, Clone, Copy)]
pub struct ProgrammingClaim<'a, S> {
    certificate: Option<ProgrammingCertificate<'a, S>>,
    reported: Option<(OptimizeTermination, S)>,
}

impl<'a, S> From<ProgrammingCertificate<'a, S>> for ProgrammingClaim<'a, S> {
    fn from(certificate: ProgrammingCertificate<'a, S>) -> Self {
        Self {
            certificate: Some(certificate),
            reported: None,
        }
    }
}

impl<'a, S: RealScalar> From<&'a ProgrammingResult<S>> for ProgrammingClaim<'a, S> {
    fn from(result: &'a ProgrammingResult<S>) -> Self {
        Self {
            certificate: result.certificate(),
            reported: Some((result.termination(), result.fun())),
        }
    }
}

/// Checks LP and QP certificates against the original problem.
///
/// The verifier recomputes residuals from the original problem data and the
/// supplied certificate; it never solves for missing multipliers.
///
/// - An optimality certificate, a point with multipliers, is checked for
///   constraint and bound feasibility, multiplier signs, stationarity of the
///   Lagrangian, complementarity, and the gap between the primal and dual
///   objectives.
/// - An infeasibility certificate, Farkas multipliers `y`, is checked for the
///   cancellation of `Aᵀy` and a positive contradiction margin.
/// - An unboundedness certificate, a feasible point and a direction `d`, is
///   checked for feasibility, for `d` being a recession direction, and for
///   descent of the objective along `d`.
///
/// For a QP with an explicit Hessian, the verifier also checks that the Hessian
/// is positive semidefinite; a QP given by a Hessian operator gets a
/// [`VerificationStatus::KktOnly`] report at best. Each test uses one role of
/// [`VerificationOptions::with_tolerances`]. The checks run in floating point:
/// an `Accepted` report means the residuals are within tolerance, not that an
/// interval proof exists. If a limit, cancellation or deadline stops the check,
/// the report is `Incomplete`.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{LpProblem, ProgrammingVerifier};
/// let problem = LpProblem::new(vec![1.0_f64])?;
/// let result = problem.solve()?;
/// assert!(result.x()[0].abs() < 1e-12);
/// let report = ProgrammingVerifier::default().verify(&problem, &result)?;
/// assert!(report.is_accepted());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgrammingVerifier<S: RealScalar> {
    options: VerificationOptions<S>,
}

impl<S: RealScalar> Default for ProgrammingVerifier<S> {
    fn default() -> Self {
        Self::new(VerificationOptions::default())
    }
}

impl<S: RealScalar> ProgrammingVerifier<S> {
    /// Creates a verifier with `options`.
    pub const fn new(options: VerificationOptions<S>) -> Self {
        Self { options }
    }
    /// Returns the options.
    pub const fn options(&self) -> &VerificationOptions<S> {
        &self.options
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> ProgrammingVerifier<S> {
    /// Creates a workspace for checks on `problem`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated, or an error from an operator's own workspace query.
    pub fn workspace<P>(&self, problem: &P) -> Result<VerificationWorkspace<S>, SciOxideError>
    where
        P: VerificationProblem<S> + ?Sized,
    {
        problem.prepare()
    }

    /// Checks `claim` against `problem` on the calling thread and returns a new report.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace`] and [`Self::verify_into`].
    pub fn verify<'c, P>(
        &self,
        problem: &P,
        claim: impl Into<ProgrammingClaim<'c, S>>,
    ) -> Result<VerificationReport<S>, SciOxideError>
    where
        P: VerificationProblem<S> + ?Sized,
        S: 'c,
    {
        let mut workspace = self.workspace(problem)?;
        let mut report = VerificationReport::empty();
        self.verify_into(
            problem,
            claim,
            &mut report,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(report)
    }

    /// Checks `claim` against `problem` and writes the report to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent solve context. Each operator application counts as one
    /// evaluation. With a workspace from [`Self::workspace`] for this problem,
    /// the check does not allocate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`, `DimensionMismatch` or `NonFinite` if the certificate
    ///   has the wrong lengths or non-finite entries, or the workspace buffers
    ///   do not match the problem's dimension, row count or operator
    ///   workspace length. These are detected before any operator is applied.
    /// - Errors returned by an operator, overflow in the residuals, or a failure
    ///   of the eigenvalue computation in the convexity check.
    pub fn verify_into<'scope, 'capability, 'c, P>(
        &self,
        problem: &P,
        claim: impl Into<ProgrammingClaim<'c, S>>,
        output: &mut VerificationReport<S>,
        workspace: &mut VerificationWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        P: VerificationProblem<S> + ?Sized,
        S: 'c,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let claim = claim.into();
        if let Some(certificate) = claim.certificate {
            problem.validate(certificate, workspace)?;
        }
        crate::attempt::preflight(&scope, None)?;
        let tolerances = self.options.tolerances;
        let identity = crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "ProgrammingVerifier", 1),
            problem.dimension(),
            &self.options,
        );
        let budgets = self
            .options
            .max_evaluations
            .map_or(SolveBudgets::unbounded(), |limit| {
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit)
            });
        let mut shell = core::mem::take(&mut workspace.shell);
        let (outcome, started) = shell.execute(
            scope,
            crate::attempt::Entry::Call { identity, budgets },
            |context| {
                let Some(certificate) = claim.certificate else {
                    return Ok(VerificationReport::empty());
                };
                let report = problem.check(certificate, tolerances, workspace, context)?;
                match claim.reported {
                    Some((termination, objective)) => {
                        reported_claim(certificate, termination, objective, report, tolerances)
                    }
                    None => Ok(report),
                }
            },
        );
        let stopped = outcome
            .as_ref()
            .is_ok_and(|report| report.status == VerificationStatus::Incomplete);
        shell.finish(
            AttemptKind::Call,
            started,
            outcome.is_err(),
            outcome.is_ok() && !stopped,
            outcome.is_ok().then_some(()),
        );
        workspace.shell = shell;
        *output = outcome?;
        Ok(())
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> VerificationProblem<S> for crate::LpProblem<S> {}
impl<S: RealScalar + NumericScalar<Real = S>> sealed::Sealed<S> for crate::LpProblem<S> {
    fn dimension(&self) -> usize {
        sealed::Sealed::dimension(&LpVerificationProblem::from_problem(self))
    }
    fn prepare(&self) -> Result<VerificationWorkspace<S>, SciOxideError> {
        sealed::Sealed::prepare(&LpVerificationProblem::from_problem(self))
    }
    fn validate(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        workspace: &VerificationWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        sealed::Sealed::validate(
            &LpVerificationProblem::from_problem(self),
            certificate,
            workspace,
        )
    }
    fn check(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        tolerances: OptimalityTolerances<S>,
        workspace: &mut VerificationWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<VerificationReport<S>, SciOxideError> {
        sealed::Sealed::check(
            &LpVerificationProblem::from_problem(self),
            certificate,
            tolerances,
            workspace,
            context,
        )
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> VerificationProblem<S> for crate::QpProblem<S> {}
impl<S: RealScalar + NumericScalar<Real = S>> sealed::Sealed<S> for crate::QpProblem<S> {
    fn dimension(&self) -> usize {
        sealed::Sealed::dimension(&QpVerificationProblem::from_problem(self))
    }
    fn prepare(&self) -> Result<VerificationWorkspace<S>, SciOxideError> {
        sealed::Sealed::prepare(&QpVerificationProblem::from_problem(self))
    }
    fn validate(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        workspace: &VerificationWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        sealed::Sealed::validate(
            &QpVerificationProblem::from_problem(self),
            certificate,
            workspace,
        )
    }
    fn check(
        &self,
        certificate: ProgrammingCertificate<'_, S>,
        tolerances: OptimalityTolerances<S>,
        workspace: &mut VerificationWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<VerificationReport<S>, SciOxideError> {
        sealed::Sealed::check(
            &QpVerificationProblem::from_problem(self),
            certificate,
            tolerances,
            workspace,
            context,
        )
    }
}
