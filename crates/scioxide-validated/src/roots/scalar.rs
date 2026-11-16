use super::*;

use crate::IntervalScalarFunction;

pub(crate) const INTERVAL_NEWTON: AlgorithmId =
    scioxide_core::algorithm_id!("validated-roots", "interval-newton", 1);

/// Proposed claim about the roots of a real function in an interval, checked
/// by [`ScalarRootVerifier::verify`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarRootCertificate<R: RealScalar> {
    /// Interval `X` to search; bounded and nonempty.
    pub region: ExtendedInterval<R>,
    /// Point `c ∈ X` for the Newton step, usually an approximate root.
    pub center: R,
    /// Claim to prove.
    pub claim: RootClaim,
}
/// Proves or refutes claims about the roots of a continuous real function `f`
/// in an interval, and isolates its roots.
///
/// `f` and its derivative `f′` are given as [`IntervalScalarFunction`]s,
/// usually closures. The function must be continuous, and the derivative
/// extension must enclose `f′`, on every interval the verifier evaluates.
///
/// # Method
///
/// For a claim about the interval `X` with center `c`, the verifier proves:
///
/// - `Empty` if `0 ∉ f(X)`, or if `f′(X)` does not contain zero and the
///   interval Newton step `N(X) = c − f(c)/f′(X)` does not meet `X`;
/// - `Exists` if `f(c) = 0`, or if `f` vanishes at an end of `X` or has
///   opposite signs at its ends (intermediate value theorem);
/// - `Unique` if, in addition, `0 ∉ f′(X)`, so that `f` is strictly monotone.
///   With `0 ∉ f′(X)`, the inclusion of `N(X)` in the interior of `X` also
///   proves `Unique`.
///
/// One verification counts as one iteration and at most five evaluations of
/// the callbacks. It does not allocate.
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// use scioxide_validated::{
///     DecoratedInterval as D, RootClaim, ScalarRootCertificate, ScalarRootVerifier, ValidatedOptions,
/// };
///
/// // f(x) = x² − 2 has exactly one root, √2, in [1, 2].
/// let two = D::point(2.0_f64)?;
/// let value = |x: D<f64>| x.square()?.sub(&two);
/// let derivative = |x: D<f64>| x.mul(&two);
/// let verifier = ScalarRootVerifier::new(&value, &derivative, ValidatedOptions::new());
/// let report = verifier.verify(ScalarRootCertificate {
///     region: ExtendedInterval::new(Some(1.0), Some(2.0))?,
///     center: 1.5,
///     claim: RootClaim::Unique,
/// })?;
/// assert!(report.is_proved());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct ScalarRootVerifier<'a, R: RealScalar> {
    value: &'a (dyn IntervalScalarFunction<R> + Sync),
    derivative: &'a (dyn IntervalScalarFunction<R> + Sync),
    options: ValidatedOptions,
}
impl<R: RealScalar> core::fmt::Debug for ScalarRootVerifier<'_, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScalarRootVerifier")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<'a, R: RealScalar> ScalarRootVerifier<'a, R> {
    /// Creates a verifier for the function `value` with derivative `derivative`.
    pub fn new(
        value: &'a (dyn IntervalScalarFunction<R> + Sync),
        derivative: &'a (dyn IntervalScalarFunction<R> + Sync),
        options: ValidatedOptions,
    ) -> Self {
        Self {
            value,
            derivative,
            options,
        }
    }
    /// Returns the options.
    pub const fn options(&self) -> &ValidatedOptions {
        &self.options
    }
    pub(super) const fn identity(&self) -> SolveIdentity {
        identity(INTERVAL_NEWTON, 1)
    }
    /// Returns the workspace size for `request`: report slots for batches, or
    /// an isolation into a tree of `request.capacity()` nodes.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(&self, request: ProofRequest) -> Result<RootRequirements> {
        let shape = workspace::Shape {
            tree_order: if request.capacity() > 0 { 1 } else { 0 },
            nodes: request.capacity(),
            ..workspace::Shape::default()
        };
        RootRequirements::new::<R>(shape, request.proof_count(), self.options.byte_limit())
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if it cannot be sized or reserved.
    pub fn workspace(&self, request: ProofRequest) -> Result<RootWorkspace<R>> {
        let mut workspace = RootWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Proves or refutes the claim of `certificate`, without allocating.
    ///
    /// A callback result that is empty, unbounded or decorated below `Dac`
    /// makes the proof inconclusive. Runs sequentially in a new solve.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the center is NaN or infinite.
    /// - `InvalidInput` if the region is empty or unbounded, the center lies
    ///   outside it, or the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - Errors of the callbacks.
    pub fn verify(
        &self,
        certificate: ScalarRootCertificate<R>,
    ) -> Result<RootVerificationReport<R>> {
        self.verify_scoped(certificate, ExecutionContext::sequential_native())
    }
    /// [`Self::verify`] run in `scope`; scalar proofs need no workspace.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify`].
    pub fn verify_scoped<'s, 'c>(
        &self,
        certificate: ScalarRootCertificate<R>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootVerificationReport<R>> {
        self.validate_certificate(certificate)?;
        crate::shell::run_local(
            scope.into(),
            self.identity(),
            self.options.budgets(),
            |context| self.verify_in(certificate, context),
        )
    }
    /// Verify inside an existing epoch, mapping stops and committing decided claims.
    pub(super) fn verify_in(
        &self,
        certificate: ScalarRootCertificate<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RootVerificationReport<R>> {
        let result = self.check(certificate, context);
        finish(result, context)
    }
    /// Verifies independent claims, writing one report per certificate.
    ///
    /// The proofs run in parallel under the options' execution policy, and the
    /// reports do not depend on the policy or the number of workers. On error,
    /// `reports` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `reports` and `certificates` differ in length.
    /// - The errors of [`Self::verify`] for any certificate, errors of the
    ///   executor, and those of [`RootWorkspace::try_reserve`] if the workspace
    ///   must grow.
    pub fn verify_batch_into<'s, 'c>(
        &self,
        certificates: &[ScalarRootCertificate<R>],
        reports: &mut [RootVerificationReport<R>],
        workspace: &mut RootWorkspace<R>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<()> {
        let count = reports.len();
        workspace
            .core
            .verify_batch(
                scope.into(),
                self.identity(),
                &self.options,
                || {
                    let requirements =
                        self.workspace_requirements(ProofRequest::proofs(certificates.len()))?;
                    crate::shell::exact(certificates.len(), count)?;
                    certificates
                        .iter()
                        .try_for_each(|&c| self.validate_certificate(c))?;
                    Ok(requirements)
                },
                reports,
                |_, _| Ok(crate::shell::Prepared::Ready(())),
                &|index, _, _, _, context| self.verify_in(certificates[index], context),
            )
            .map(|_| ())
    }
    fn validate_certificate(&self, c: ScalarRootCertificate<R>) -> Result<()> {
        crate::environment::check()?;
        if !c.center.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "scalar root center must be finite",
            ));
        }
        if c.region.is_empty() || !c.region.is_finite() || !c.region.contains(c.center) {
            return Err(invalid(
                scioxide_core::argument!("certificate"),
                "invalid scalar root proposal",
            ));
        }
        Ok(())
    }
    /// Scalar proof of the `certificate` argument.
    fn check(
        &self,
        c: ScalarRootCertificate<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RootVerificationReport<R>> {
        const REGION: Argument = scioxide_core::argument!("certificate.region");
        const PLAN: Argument = scioxide_core::argument!("self");
        const DOMAIN: RootInconclusive = RootInconclusive::Reason(RootProofFailure::Domain);
        crate::environment::check()?;
        if let Some(reason) = iteration(context)? {
            return Ok(RootVerificationReport::unknown(reason));
        }
        macro_rules! eval {
            ($f:expr,$x:expr,$role:expr) => {{
                if let Some(reason) = admit(context, $role)? {
                    return Ok(RootVerificationReport::unknown(reason));
                }
                let result = $f.enclose_scoped($x, context)?;
                crate::environment::check()?;
                // An empty enclosure, like a decoration below `Dac`, asserts no
                // continuity on the input.
                if result.set().is_empty() {
                    return Ok(RootVerificationReport::unknown(DOMAIN));
                }
                if !result.set().is_finite() {
                    return Ok(RootVerificationReport::unknown(RootInconclusive::Unbounded));
                }
                if result.decoration() < Decoration::Dac {
                    return Ok(RootVerificationReport::unknown(DOMAIN));
                }
                result
            }};
        }
        let x = D::from(c.region);
        let fx = eval!(self.value, x, EvaluationRole::Function);
        if !fx.contains(R::ZERO) {
            return Ok(RootVerificationReport::fact(c.claim, RootClaim::Empty));
        }
        let fc = eval!(self.value, D::point(c.center)?, EvaluationRole::Function);
        let derivative = eval!(self.derivative, x, EvaluationRole::Jacobian);
        let monotone = !derivative.contains(R::ZERO);
        let mut exists = fc.bounds() == Some((R::ZERO, R::ZERO));
        if monotone {
            let newton = D::point(c.center)?.sub_admitted(&fc.div_admitted(&derivative)?)?;
            if newton.set().intersection(&c.region).is_empty() {
                return Ok(RootVerificationReport::fact(c.claim, RootClaim::Empty));
            }
            // Strict inclusion is the interval Newton existence theorem. Closed
            // boundary cases are decided independently by endpoint signs below.
            let (lo, hi) = bounded(x, REGION)?;
            if newton.set().is_finite() {
                let (nl, nh) = bounded(newton, PLAN)?;
                exists |= nl > lo && nh < hi;
            }
        }
        if !exists {
            let (lo, hi) = bounded(x, REGION)?;
            let left = eval!(self.value, D::point(lo)?, EvaluationRole::Function);
            let right = eval!(self.value, D::point(hi)?, EvaluationRole::Function);
            let (ll, lh) = bounded(left, PLAN)?;
            let (rl, rh) = bounded(right, PLAN)?;
            exists = (ll == R::ZERO && lh == R::ZERO)
                || (rl == R::ZERO && rh == R::ZERO)
                || (lh < R::ZERO && rl > R::ZERO)
                || (rh < R::ZERO && ll > R::ZERO);
        }
        Ok(if exists {
            RootVerificationReport::fact(
                c.claim,
                if monotone {
                    RootClaim::Unique
                } else {
                    RootClaim::Exists
                },
            )
        } else {
            RootVerificationReport::unknown(RootInconclusive::Reason(RootProofFailure::Inequality))
        })
    }
}
