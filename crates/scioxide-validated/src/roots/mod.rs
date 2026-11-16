//! Verified existence, uniqueness and absence of roots.
mod isolation;
pub(crate) mod polynomial;
pub(crate) mod scalar;
mod system;
#[cfg(test)]
mod tests;
mod workspace;

pub use isolation::{RootIsolation, RootIsolationNode, RootIsolationReport, RootIsolationView};
pub use polynomial::{
    PolynomialIsolationReport, PolynomialRootCertificate, PolynomialRootCover, PolynomialRootPlan,
    RootDisk,
};
pub use scalar::{ScalarRootCertificate, ScalarRootVerifier};
pub use system::{
    SystemRootCertificate, SystemRootCertificateView, SystemRootCertificateViewMut,
    SystemRootVerifier,
};
pub use workspace::{RootRequirements, RootWorkspace};

use crate::shell::{Ledger, Outcome, invalid};
use crate::{
    DecoratedInterval as D, Decoration, interval_math::*, point_enclosure, rectangle::Box2,
};
use crate::{IntervalBox, ProofRequest, ValidatedOptions};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AlgorithmId, Argument, Complex, EvaluationRole, ExecutionContext, ExecutionStopReason,
    ExtendedInterval, NumericScalar, PartialResultTrust, ProblemToken, RealScalar, SciOxideError,
    SolveContext, SolveIdentity, SolveScope,
};

type Result<T> = core::result::Result<T, SciOxideError>;

/// Claim about the roots of a function in a region, to be proved by
/// [`ScalarRootVerifier`] or [`SystemRootVerifier`].
///
/// Polynomial root counts use [`PolynomialRootCertificate`] instead. A proof
/// of `Unique` also proves `Exists`, and a proof of either refutes `Empty`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootClaim {
    /// The region contains at least one root.
    Exists,
    /// The region contains exactly one root.
    Unique,
    /// The region contains no root.
    Empty,
}
/// Reason specific to root proofs why a proof is inconclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootProofFailure {
    /// The tests of the method decide neither the claim nor its opposite, for
    /// example because the region is too wide.
    Inequality,
    /// A callback returned an empty enclosure or a decoration below `Dac`, so
    /// the function is not known to be continuous on the region.
    Domain,
}
/// Reason why a root proof is inconclusive.
pub type RootInconclusive = crate::ProofInconclusive<RootProofFailure>;

/// Termination status of root isolation and polynomial cover computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootIsolationTermination {
    /// Every region is decided, and the regions account for every root.
    Complete,
    /// A proof could not be completed, or execution stopped.
    Inconclusive(RootInconclusive),
    /// No further subdivision or disk growth is possible.
    Limit(crate::RefinementLimit),
    /// Verification of a supplied tree refuted the claim of a leaf. The
    /// report's node count is the index of that leaf.
    Refuted,
}
impl From<RootInconclusive> for RootIsolationTermination {
    fn from(reason: RootInconclusive) -> Self {
        Self::Inconclusive(reason)
    }
}
/// Outcome of a root proof.
///
/// A claim that is not proved is not thereby refuted: only `Refuted` shows that
/// it is false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootVerificationStatus {
    /// The report slot has not been filled; no proof has run.
    NotRun,
    /// The claim is proved.
    Proved,
    /// The claim is proved false.
    Refuted,
    /// Neither the claim nor its opposite was proved.
    Inconclusive(RootInconclusive),
}
/// Status and diagnostics of a scalar or system root proof.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RootVerificationReport<R: RealScalar> {
    status: RootVerificationStatus,
    evidence: Option<RootClaim>,
    defect: Option<R>,
}
impl<R: RealScalar> RootVerificationReport<R> {
    /// Returns an empty report with status `NotRun`, for the report slots of
    /// the `verify_batch_into` methods.
    pub const fn not_run() -> Self {
        Self {
            status: RootVerificationStatus::NotRun,
            evidence: None,
            defect: None,
        }
    }
    /// Returns the status of the claim.
    pub const fn status(&self) -> RootVerificationStatus {
        self.status
    }
    /// Returns the strongest claim proved during this call, whatever was
    /// claimed, or `None`.
    pub const fn evidence(&self) -> Option<RootClaim> {
        self.evidence
    }
    /// Returns the upper bound on `‖I − R·J(X)‖` of the Krawczyk test of
    /// [`SystemRootVerifier`], when it was computed.
    pub const fn defect(&self) -> Option<R> {
        self.defect
    }
    /// Returns whether the status is `Proved`.
    pub fn is_proved(&self) -> bool {
        self.status == RootVerificationStatus::Proved
    }
    const fn unknown(reason: RootInconclusive) -> Self {
        Self {
            status: RootVerificationStatus::Inconclusive(reason),
            evidence: None,
            defect: None,
        }
    }
    /// Decide `claim` from the established `fact`.
    const fn fact(claim: RootClaim, fact: RootClaim) -> Self {
        use RootClaim::{Empty, Exists, Unique};
        let status = match (claim, fact) {
            (Exists, Exists | Unique) | (Unique, Unique) | (Empty, Empty) => {
                RootVerificationStatus::Proved
            }
            (Exists | Unique, Empty) | (Empty, Exists | Unique) => RootVerificationStatus::Refuted,
            (Unique, Exists) => RootVerificationStatus::Inconclusive(RootInconclusive::Reason(
                RootProofFailure::Inequality,
            )),
        };
        Self {
            status,
            evidence: Some(fact),
            defect: None,
        }
    }
}
/// Status and diagnostics of a polynomial root count.
///
/// Counts include multiplicities. A claimed count is refuted only when a
/// different count is proved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct PolynomialVerificationReport {
    status: RootVerificationStatus,
    count: Option<usize>,
}
impl PolynomialVerificationReport {
    /// Returns an empty report with status `NotRun`, for the report slots of
    /// [`PolynomialRootPlan::verify_batch_into`].
    pub const fn not_run() -> Self {
        Self {
            status: RootVerificationStatus::NotRun,
            count: None,
        }
    }
    /// Returns the status of the claimed count.
    pub const fn status(&self) -> RootVerificationStatus {
        self.status
    }
    /// Returns the number of roots proved to lie in the disk, counted with
    /// multiplicity, whatever count was claimed.
    pub const fn count(&self) -> Option<usize> {
        self.count
    }
    /// Returns whether the status is `Proved`.
    pub fn is_proved(&self) -> bool {
        self.status == RootVerificationStatus::Proved
    }
    const fn unknown(reason: RootInconclusive) -> Self {
        Self {
            status: RootVerificationStatus::Inconclusive(reason),
            count: None,
        }
    }
    /// Decide the `claim`ed count from an established `count`.
    const fn counted(claim: usize, count: usize) -> Self {
        Self {
            status: if claim == count {
                RootVerificationStatus::Proved
            } else {
                RootVerificationStatus::Refuted
            },
            count: Some(count),
        }
    }
}
/// A root report that `finish` maps and commits.
trait Report: Outcome {
    /// Status of the requested claim or count.
    fn status(&self) -> RootVerificationStatus;
}
impl<R: RealScalar> Outcome for RootVerificationReport<R> {
    fn not_run() -> Self {
        Self::not_run()
    }
    fn stopped(reason: ExecutionStopReason) -> Self {
        Self::unknown(RootInconclusive::Stopped(reason))
    }
    fn succeeded(&self) -> bool {
        self.is_proved()
    }
}
impl<R: RealScalar> Report for RootVerificationReport<R> {
    fn status(&self) -> RootVerificationStatus {
        self.status
    }
}
impl Outcome for PolynomialVerificationReport {
    fn not_run() -> Self {
        Self::not_run()
    }
    fn stopped(reason: ExecutionStopReason) -> Self {
        Self::unknown(RootInconclusive::Stopped(reason))
    }
    fn succeeded(&self) -> bool {
        self.is_proved()
    }
}
impl Report for PolynomialVerificationReport {
    fn status(&self) -> RootVerificationStatus {
        self.status
    }
}
/// Batch ledger slot shared by every root plan; each report reads back only its fields.
#[derive(Debug, Clone, Copy)]
struct Staged<R: RealScalar> {
    status: RootVerificationStatus,
    evidence: Option<RootClaim>,
    count: Option<usize>,
    defect: Option<R>,
}
impl<R: RealScalar> Outcome for Staged<R> {
    fn not_run() -> Self {
        RootVerificationReport::not_run().into_staged()
    }
    fn stopped(reason: ExecutionStopReason) -> Self {
        <RootVerificationReport<R> as Outcome>::stopped(reason).into_staged()
    }
    fn succeeded(&self) -> bool {
        self.status == RootVerificationStatus::Proved
    }
}
impl<R: RealScalar> RootVerificationReport<R> {
    const fn into_staged(self) -> Staged<R> {
        Staged {
            status: self.status,
            evidence: self.evidence,
            count: None,
            defect: self.defect,
        }
    }
}
impl<R: RealScalar> Ledger<RootVerificationReport<R>> for Staged<R> {
    fn stage(report: RootVerificationReport<R>) -> Self {
        report.into_staged()
    }
    fn report(self) -> RootVerificationReport<R> {
        RootVerificationReport {
            status: self.status,
            evidence: self.evidence,
            defect: self.defect,
        }
    }
}
impl<R: RealScalar> Ledger<PolynomialVerificationReport> for Staged<R> {
    fn stage(report: PolynomialVerificationReport) -> Self {
        Self {
            status: report.status,
            evidence: None,
            count: report.count,
            defect: None,
        }
    }
    fn report(self) -> PolynomialVerificationReport {
        PolynomialVerificationReport {
            status: self.status,
            count: self.count,
        }
    }
}
/// Pending stop.
fn stop(context: &SolveContext<'_, '_>) -> Result<Option<RootInconclusive>> {
    Ok(context.stop_reason()?.map(RootInconclusive::Stopped))
}
/// Charge one provider call to the budget.
fn admit(
    context: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
) -> Result<Option<RootInconclusive>> {
    if let Some(reason) = stop(context)? {
        return Ok(Some(reason));
    }
    Ok((!context.record_provider(role)?.is_granted()).then_some(RootInconclusive::EvaluationLimit))
}
/// The inconclusive reason of a provider invocation that did not evaluate.
fn invoked(invocation: scioxide_core::Invocation<()>) -> Option<RootInconclusive> {
    match invocation {
        scioxide_core::Invocation::Evaluated(()) => None,
        scioxide_core::Invocation::BudgetExhausted(_) => Some(RootInconclusive::EvaluationLimit),
        scioxide_core::Invocation::ExecutionStopped(reason) => {
            Some(RootInconclusive::Stopped(reason))
        }
    }
}
/// Charge one iteration to the budget.
fn iteration(context: &mut SolveContext<'_, '_>) -> Result<Option<RootInconclusive>> {
    if let Some(reason) = stop(context)? {
        return Ok(Some(reason));
    }
    Ok((!context.record_iteration()?.is_granted()).then_some(RootInconclusive::IterationLimit))
}
/// Map stops and commit decided claims.
fn finish<T: Report>(result: Result<T>, context: &mut SolveContext<'_, '_>) -> Result<T> {
    let report = match result {
        Ok(report) => report,
        Err(error) => return crate::shell::stopped(&error).map(T::stopped).ok_or(error),
    };
    if let Some(reason) = context.stop_reason()? {
        return Ok(T::stopped(reason));
    }
    if matches!(
        report.status(),
        RootVerificationStatus::Proved | RootVerificationStatus::Refuted
    ) {
        context.commit_accepted(PartialResultTrust::CertificateVerified)?;
    }
    Ok(report)
}
/// The identity of a root proof by `algorithm` over a problem of `shape` components.
const fn identity(algorithm: AlgorithmId, shape: usize) -> SolveIdentity {
    SolveIdentity::new(ProblemToken::new(0), algorithm, shape as u128, 0)
}
fn disjoint<R: RealScalar>(a: IntervalBox<R>, b: IntervalBox<R>) -> bool {
    a.re.intersection(&b.re).is_empty() || a.im.intersection(&b.im).is_empty()
}
