//! Certified integration over an interval.
mod adaptive;
mod gauss;
mod proof;
mod scheduler;
#[cfg(test)]
mod tests;
mod workspace;

use crate::shell::{Outcome, breakdown, invalid};
use crate::{DecoratedInterval as D, interval_math::*, rectangle::Box2};
use crate::{IntervalBox, ProofRequest, ValidatedOptions};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    AlgorithmId, Argument, ErrorTolerance, EvaluationRole, ExecutionContext, ExecutionStopReason,
    ExtendedInterval, NumericScalar, PartialResultTrust, ProblemToken, Quantity, RealScalar,
    SciOxideError, SolveContext, SolveIdentity, SolveScope,
};
pub use workspace::{QuadratureRequirements, QuadratureWorkspace};

type Result<T> = core::result::Result<T, SciOxideError>;

/// What you assert about the integrand `f` on a piece of the partition, which
/// selects how the piece's integral is bounded.
///
/// Each assumption is your promise, together with the enclosures your
/// [`QuadratureOracle`] returns; the verifier does not check it. On a panel
/// `[a, b]` of width `h = b − a`, the bounds are, separately for the real and
/// imaginary parts of each component:
///
/// ```text
/// IntegrableRange:                       ∫ f ∈ h · f([a, b])
/// TwiceContinuouslyDifferentiable:       ∫ f ∈ (h/2)(f(a) + f(b)) − (h³/12) f″([a, b])
/// EightTimesContinuouslyDifferentiable:  ∫ f ∈ G₄(f) + (h⁹/1778112000) f⁽⁸⁾([a, b])
/// ```
///
/// where `f([a, b])` is the oracle's enclosure of `f` on the panel and `G₄` is
/// the four-point Gauss–Legendre rule, exact for polynomials of degree up to
/// seven. These panels are bisected during generation. On the other two kinds
/// of piece the oracle encloses the integral directly, and the piece is never
/// subdivided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureAssumption {
    /// `f` is integrable and bounded on a finite piece; the oracle encloses its
    /// values. One query per panel.
    IntegrableRange,
    /// `f` is twice continuously differentiable on a finite piece; the oracle
    /// encloses its values at the panel ends and its second derivative on the
    /// panel. Three queries per panel.
    TwiceContinuouslyDifferentiable,
    /// `f` is eight times continuously differentiable on a finite piece; the
    /// oracle encloses its values on narrow intervals around the four Gauss
    /// nodes and its eighth derivative on the panel. Five queries per panel.
    ///
    /// # References
    ///
    /// NIST DLMF, §3.5(v), Gauss quadrature and its remainder.
    /// <https://dlmf.nist.gov/3.5.v>
    EightTimesContinuouslyDifferentiable,
    /// `f` is integrable on a finite piece, possibly with a singularity; the
    /// oracle encloses the integral over the whole piece.
    IntegrableSingularity,
    /// The integral over a piece with one infinite end converges; the oracle
    /// encloses it.
    ConvergentTail,
}
impl QuadratureAssumption {
    fn regular(self) -> bool {
        matches!(
            self,
            Self::IntegrableRange
                | Self::TwiceContinuouslyDifferentiable
                | Self::EightTimesContinuouslyDifferentiable
        )
    }
}
/// Piece of the partition passed to [`QuadratureVerifier::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePiece<R: RealScalar> {
    /// Interval of the piece; an absent end is infinite.
    pub domain: ExtendedInterval<R>,
    /// What you assert about the integrand on this piece.
    pub assumption: QuadratureAssumption,
}
/// Panel of a quadrature certificate: a subinterval of one piece, which
/// carries that piece's assumption.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePanel<R: RealScalar> {
    /// Index of the piece in the partition of the verifier.
    pub piece: usize,
    /// Interval of the panel.
    pub domain: ExtendedInterval<R>,
}
/// Quantity that a [`QuadratureRequest`] asks the oracle to enclose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureQuery {
    /// `{f(x) : x ∈ domain}`; the domain may be a single point or a narrow
    /// interval around one.
    Value,
    /// `{f″(x) : x ∈ domain}`.
    SecondDerivative,
    /// `{f⁽⁸⁾(x) : x ∈ domain}`.
    EighthDerivative,
    /// `∫ f(x) dx` over the domain.
    Integral,
}
/// Request passed to [`QuadratureOracle::enclose_scoped`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadratureRequest<R: RealScalar> {
    /// Index of the piece that contains `domain`.
    pub piece: usize,
    /// Interval of the query: a point, a panel, or an infinite tail.
    pub domain: ExtendedInterval<R>,
    /// Quantity to enclose.
    pub query: QuadratureQuery,
}
/// Whether the oracle could enclose the requested quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureObligation {
    /// Every component of the output encloses the requested quantity.
    Established,
    /// The oracle cannot enclose it; the output is ignored and the proof is
    /// inconclusive.
    Unavailable,
}
/// Source of enclosures of the integrand `f`, its derivatives and its
/// integrals, for [`QuadratureVerifier`].
///
/// For each [`QuadratureRequest`], write an enclosure of the requested
/// quantity for every component of `f` to `output` and return `Established`,
/// or return `Unavailable`. The enclosures must include the rounding errors of
/// their evaluation, and imaginary parts must be exactly zero for real
/// problems. The [`QuadratureAssumption`] of each piece is your promise; point
/// samples of `f` do not meet it. A verifier cannot detect an oracle that
/// breaks these rules; see
/// [Interval extensions you provide](crate#interval-extensions-you-provide).
/// The verifier counts each request toward the evaluation budget before it makes
/// it; override `enclose_scoped` to charge nested work to its `context`. Errors you return
/// propagate unchanged.
///
/// Closures `Fn(QuadratureRequest<R>, &mut [IntervalBox<R>]) -> Result<QuadratureObligation, SciOxideError>`
/// implement this trait.
///
/// ```
/// use scioxide_validated::{
///     IntervalBox, QuadratureObligation, QuadratureOracle, QuadratureRequest, point_enclosure,
/// };
///
/// // f(x) = 2, enclosed exactly on every interval.
/// let constant =
///     |_: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
///         out.fill(point_enclosure(2.0_f64)?);
///         Ok(QuadratureObligation::Established)
///     };
/// fn takes(_: &dyn QuadratureOracle<f64>) {}
/// takes(&constant);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait QuadratureOracle<R: RealScalar> {
    /// Encloses the requested quantity using the mathematical inputs.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn enclose(
        &self,
        request: QuadratureRequest<R>,
        output: &mut [IntervalBox<R>],
    ) -> Result<QuadratureObligation>;

    /// Encloses the requested quantity inside the caller's solve.
    ///
    /// Override this method to charge nested work to `context`.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn enclose_scoped(
        &self,
        request: QuadratureRequest<R>,
        output: &mut [IntervalBox<R>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<QuadratureObligation> {
        let _ = context;
        self.enclose(request, output)
    }
}
impl<R, F> QuadratureOracle<R> for F
where
    R: RealScalar,
    F: Fn(QuadratureRequest<R>, &mut [IntervalBox<R>]) -> Result<QuadratureObligation> + ?Sized,
{
    fn enclose(
        &self,
        request: QuadratureRequest<R>,
        output: &mut [IntervalBox<R>],
    ) -> Result<QuadratureObligation> {
        self(request, output)
    }
}

/// Reason specific to quadrature why a proof is inconclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureProofFailure {
    /// The oracle returned `Unavailable` for a request the proof needs.
    MissingObligation,
}
/// Reason why a quadrature proof is inconclusive.
pub type QuadratureInconclusive = crate::ProofInconclusive<QuadratureProofFailure>;
/// Reason why [`QuadratureVerifier::verify`] rejects a certificate.
///
/// A rejection means that the recorded bounds are narrower than the bounds the
/// verifier recomputes. It does not mean that the integral lies outside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureRejection {
    /// The recorded bound of a panel does not contain the recomputed bound.
    LocalBound,
    /// The recorded total does not contain the sum of the recorded panel
    /// bounds.
    TotalBound,
}
/// Outcome of [`QuadratureVerifier::verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureVerificationStatus {
    /// The status slot has not been filled; no check has run.
    NotRun,
    /// The panels cover the pieces, the recorded bound of every panel contains
    /// the recomputed bound of its integral, and the total contains the sum of
    /// the panel bounds rounded outward. The total encloses the integral.
    Verified,
    /// A recorded bound is too narrow.
    Rejected(QuadratureRejection),
    /// The check could not be completed.
    Inconclusive(QuadratureInconclusive),
}
/// Termination status of [`QuadratureVerifier::generate_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureTermination {
    /// The radius of the total meets the tolerance.
    ToleranceReached,
    /// Generation stopped for the given reason.
    Inconclusive(QuadratureInconclusive),
    /// No further bisection is possible.
    Limit(crate::RefinementLimit),
    /// Bisection cannot reduce the radius further: every panel that can be
    /// bisected has a point bound, so the remaining radius comes from the
    /// singular and tail pieces and from rounding the total.
    FixedBounds,
}
impl Outcome for QuadratureVerificationStatus {
    fn not_run() -> Self {
        Self::NotRun
    }
    fn stopped(reason: ExecutionStopReason) -> Self {
        Self::Inconclusive(QuadratureInconclusive::Stopped(reason))
    }
    fn succeeded(&self) -> bool {
        *self == Self::Verified
    }
}
/// Summary of a quadrature generation.
///
/// The generated certificate is a proposal; pass it to
/// [`QuadratureVerifier::verify`] to certify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct QuadratureGenerationReport {
    termination: QuadratureTermination,
    panels: usize,
}
impl QuadratureGenerationReport {
    /// Returns the termination status.
    pub const fn termination(&self) -> QuadratureTermination {
        self.termination
    }
    /// Returns the number of panels written; zero if nothing was written.
    pub const fn panels(&self) -> usize {
        self.panels
    }
    /// Returns whether a certificate covering the whole domain was written.
    pub const fn has_enclosure(&self) -> bool {
        self.panels > 0
    }
}

/// Borrowed quadrature certificate: panels, a bound for each panel and a
/// total, for `c` integrand components.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureCertificateView<'a, R: RealScalar> {
    /// Panels in increasing order, covering the domain without gaps.
    pub panels: &'a [QuadraturePanel<R>],
    /// Bound of each panel integral, `c` per panel: entry `p · c + k` is
    /// component `k` of panel `p`.
    pub local_bounds: &'a [IntervalBox<R>],
    /// Enclosure of the integral, one per component.
    pub total: &'a [IntervalBox<R>],
}
/// Output of quadrature generation, with room for a fixed number of panels.
///
/// The capacity limits how far generation bisects. Generation writes a
/// certificate that covers the whole domain, or leaves the output unchanged.
/// The storage proves nothing by itself: pass [`Self::view`] to
/// [`QuadratureVerifier::verify`]. Equality and `Debug` compare the
/// capacity, the component count and [`Self::view`].
///
/// ```
/// use scioxide_validated::QuadratureCertificate;
///
/// let certificate = QuadratureCertificate::<f64>::new(8, 2)?;
/// assert_eq!((certificate.capacity(), certificate.components()), (8, 2));
/// assert!(certificate.view().panels.is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct QuadratureCertificate<R: RealScalar> {
    panels: Vec<QuadraturePanel<R>>,
    local_bounds: Vec<IntervalBox<R>>,
    total: Vec<IntervalBox<R>>,
    count: usize,
}
impl<R: RealScalar> PartialEq for QuadratureCertificate<R> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (self.view(), other.view());
        self.capacity() == other.capacity()
            && a.panels == b.panels
            && a.local_bounds == b.local_bounds
            && a.total == b.total
    }
}
impl<R: RealScalar> core::fmt::Debug for QuadratureCertificate<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let view = self.view();
        f.debug_struct("QuadratureCertificate")
            .field("capacity", &self.capacity())
            .field("components", &self.components())
            .field("panels", &view.panels)
            .field("local_bounds", &view.local_bounds)
            .field("total", &view.total)
            .finish()
    }
}
fn placeholder<R: RealScalar>() -> QuadraturePanel<R> {
    QuadraturePanel {
        piece: 0,
        domain: ExtendedInterval::empty(),
    }
}
impl<R: RealScalar> QuadratureCertificate<R> {
    /// Allocates an empty certificate with room for `max_panels` panels of
    /// `components` components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or reserved.
    pub fn new(max_panels: usize, components: usize) -> Result<Self> {
        const LOCAL_BOUNDS: Quantity = scioxide_core::quantity!("local_bounds");
        let local_bounds = checked_mul(max_panels, components, LOCAL_BOUNDS)?;
        Ok(Self {
            panels: try_filled(
                max_panels,
                placeholder(),
                scioxide_core::quantity!("panels"),
            )?,
            local_bounds: try_filled(local_bounds, sentinel(), LOCAL_BOUNDS)?,
            total: try_filled(components, sentinel(), scioxide_core::quantity!("total"))?,
            count: 0,
        })
    }
    /// Returns the largest number of panels.
    pub fn capacity(&self) -> usize {
        self.panels.len()
    }
    /// Returns the number of integrand components.
    pub fn components(&self) -> usize {
        self.total.len()
    }
    /// Returns the certificate written by the last generation; a new
    /// certificate has no panels.
    pub fn view(&self) -> QuadratureCertificateView<'_, R> {
        QuadratureCertificateView {
            panels: &self.panels[..self.count],
            local_bounds: &self.local_bounds[..self.count * self.total.len()],
            total: &self.total,
        }
    }
    /// Write a staged proposal that fits this output.
    fn publish(&mut self, staged: QuadratureCertificateView<'_, R>) {
        self.panels[..staged.panels.len()].copy_from_slice(staged.panels);
        self.local_bounds[..staged.local_bounds.len()].copy_from_slice(staged.local_bounds);
        self.total.copy_from_slice(staged.total);
        self.count = staged.panels.len();
    }
}
pub(crate) const GENERATE: AlgorithmId =
    scioxide_core::algorithm_id!("validated-quadrature", "generate", 1);
pub(crate) const VERIFY: AlgorithmId =
    scioxide_core::algorithm_id!("validated-quadrature", "verify", 1);

/// Encloses the integral `∫ₐᵇ f(x) dx` of a real- or complex-valued function
/// with `c` components over a finite or infinite interval.
///
/// You split the domain into pieces, give each piece a
/// [`QuadratureAssumption`], and provide a [`QuadratureOracle`] that encloses
/// `f`, its derivatives or its integrals. [`Self::generate`] bisects the
/// panels of the regular pieces adaptively until the total meets a tolerance,
/// and returns a certificate: a bound for each panel and their sum.
/// [`Self::verify`] checks a certificate against the oracle, recomputing every
/// panel bound, and only a `Verified` status proves that the total encloses
/// the integral. Each panel bound counts as one iteration, so verifying `N`
/// panels takes `N` iterations.
///
/// The error estimates of non-validated quadrature, such as the difference
/// between two rules, are not bounds and cannot serve as an oracle.
///
/// ```
/// use scioxide_core::{ErrorTolerance, ExecutionContext, NormSelection, SolveContext};
/// use scioxide_validated::{
///     QuadratureVerifier, DecoratedInterval as D, ProofRequest, QuadratureAssumption,
///     QuadratureCertificate, QuadratureObligation, QuadraturePiece, QuadratureVerificationStatus,
///     ValidatedOptions, point_enclosure,
/// };
///
/// // ∫₀¹ 2 dx = 2, with f enclosed exactly on every interval.
/// let domain = D::new(0_f64, 1.)?.set();
/// let pieces = [QuadraturePiece {
///     domain,
///     assumption: QuadratureAssumption::IntegrableRange,
/// }];
/// let oracle = |_, out: &mut [_]| {
///     out.fill(point_enclosure(2_f64)?);
///     Ok(QuadratureObligation::Established)
/// };
/// let verifier =
///     QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, ValidatedOptions::new())?;
/// let tolerance = ErrorTolerance::scalar(0., 0., NormSelection::Infinity)?;
/// let (report, certificate) = verifier.generate(&tolerance, 1)?;
/// let certificate = certificate.expect("covers the domain");
/// assert!(report.has_enclosure());
/// assert_eq!(
///     verifier.verify(certificate.view())?,
///     QuadratureVerificationStatus::Verified
/// );
/// // With a workspace, later calls do not allocate.
/// let mut out = QuadratureCertificate::new(4, 1)?;
/// let mut workspace = verifier.workspace(ProofRequest::refinement(out.capacity()))?;
/// verifier.generate_into(
///     &tolerance,
///     &mut out,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(out.view().total[0].re.contains(2.));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct QuadratureVerifier<'a, S: NumericScalar> {
    domain: ExtendedInterval<S::Real>,
    pieces: &'a [QuadraturePiece<S::Real>],
    components: usize,
    oracle: &'a (dyn QuadratureOracle<S::Real> + Sync),
    options: ValidatedOptions,
    marker: PhantomData<S>,
}
impl<S: NumericScalar> core::fmt::Debug for QuadratureVerifier<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuadratureVerifier")
            .field("domain", &self.domain)
            .field("pieces", &self.pieces)
            .field("components", &self.components)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<'a, S: NumericScalar> QuadratureVerifier<'a, S> {
    /// Creates a verifier for the integral of the `components`-component function
    /// that `oracle` describes over the hull of `pieces`.
    ///
    /// The pieces must be in increasing order, each starting where the previous
    /// one ends; their hull defines the domain. A piece with an infinite
    /// end must be a `ConvergentTail`, and a `ConvergentTail` must have exactly
    /// one infinite end. A domain of zero width needs a single piece with a
    /// regular assumption.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` with argument `pieces` if the partition is empty or the pieces
    ///   do not form such a partition or do not match their assumptions.
    pub fn new(
        pieces: &'a [QuadraturePiece<S::Real>],
        components: usize,
        oracle: &'a (dyn QuadratureOracle<S::Real> + Sync),
        options: ValidatedOptions,
    ) -> Result<Self> {
        let domain = proof::validate_partition(pieces)?;
        Ok(Self {
            domain,
            pieces,
            components,
            oracle,
            options,
            marker: PhantomData,
        })
    }
    /// Returns the number `c` of integrand components.
    pub const fn components(&self) -> usize {
        self.components
    }
    /// Returns the options.
    pub const fn options(&self) -> &ValidatedOptions {
        &self.options
    }
    const fn identity(&self, algorithm: AlgorithmId) -> SolveIdentity {
        SolveIdentity::new(
            ProblemToken::new(0),
            algorithm,
            self.components as u128,
            self.pieces.len() as u128,
        )
    }
    /// Returns the workspace size for `request`: batches of verifications, or
    /// one generation into a certificate of `request.capacity()` panels.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(&self, request: ProofRequest) -> Result<QuadratureRequirements> {
        QuadratureRequirements::new::<S::Real>(
            self.components,
            request.capacity(),
            request.proof_count(),
            self.options.byte_limit(),
        )
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if it cannot be sized or reserved.
    pub fn workspace(&self, request: ProofRequest) -> Result<QuadratureWorkspace<S::Real>> {
        let mut workspace = QuadratureWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
}
/// Pending stop.
fn stop(context: &SolveContext<'_, '_>) -> Result<Option<QuadratureInconclusive>> {
    Ok(context.stop_reason()?.map(QuadratureInconclusive::Stopped))
}
/// Partition validation makes every regular piece, and so every panel of one, bounded.
const UNBOUNDED_PANEL: &str = "regular quadrature panel is unbounded";
/// Finite endpoints of `interval`, which an internal invariant keeps bounded.
fn finite_endpoints<R: RealScalar>(
    interval: ExtendedInterval<R>,
    message: &'static str,
) -> Result<(R, R)> {
    match interval.bounds() {
        Some((Some(a), Some(b))) => Ok((a, b)),
        _ => Err(breakdown(message)),
    }
}
/// Charge one iteration to the budget.
fn iteration(context: &mut SolveContext<'_, '_>) -> Result<Option<QuadratureInconclusive>> {
    if let Some(reason) = stop(context)? {
        return Ok(Some(reason));
    }
    Ok((!context.record_iteration()?.is_granted())
        .then_some(QuadratureInconclusive::IterationLimit))
}
