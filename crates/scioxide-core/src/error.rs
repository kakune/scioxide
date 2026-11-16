//! The error type shared by SciOxide crates and its classification.
//!
//! The main kinds do not overlap. `InvalidInput` is a structural problem with
//! an argument, `DomainError` a finite argument outside the mathematical
//! domain, `NonFinite` a NaN or infinity, `NumericalInstability` a breakdown
//! of finite-precision arithmetic, and `ConvergenceFailure` an iteration on
//! valid finite input that did not converge. See [errors](crate#errors) and
//! [non-finite values](crate#non-finite-values).

use crate::{Argument, ConversionType, Quantity};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::fmt::Display as _;

mod diagnostics;
pub use diagnostics::{ConvergenceDiagnostics, ConvergenceDiagnosticsView, NestedLevel};

/// Array shape reported by a `DimensionMismatch` error.
///
/// Shapes of rank at most two are stored without allocation. Vectors and
/// boxed slices convert with [`From`] without copying their axes.
///
/// ```
/// use scioxide_core::Dimensions;
///
/// let dimensions = Dimensions::from(vec![2, 3, 4]);
/// assert_eq!(dimensions.axes(), &[2, 3, 4]);
/// assert_eq!(dimensions.rank(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct Dimensions(DimensionStorage);
#[derive(Debug, Clone, PartialEq, Eq)]
enum DimensionStorage {
    Inline { axes: [usize; 2], rank: InlineRank },
    Dynamic(Vec<usize>),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineRank {
    Scalar,
    Vector,
    Matrix,
}
impl InlineRank {
    const fn len(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Vector => 1,
            Self::Matrix => 2,
        }
    }
}
impl Dimensions {
    /// Returns the rank-zero shape of a scalar.
    pub const fn scalar() -> Self {
        Self(DimensionStorage::Inline {
            axes: [0, 0],
            rank: InlineRank::Scalar,
        })
    }
    /// Returns the shape of a vector of `length` elements.
    pub const fn one(length: usize) -> Self {
        Self(DimensionStorage::Inline {
            axes: [length, 0],
            rank: InlineRank::Vector,
        })
    }
    /// Returns the shape of a `first` × `second` matrix.
    pub const fn two(first: usize, second: usize) -> Self {
        Self(DimensionStorage::Inline {
            axes: [first, second],
            rank: InlineRank::Matrix,
        })
    }
    /// Returns the number of axes.
    pub fn rank(&self) -> usize {
        self.axes().len()
    }
    /// Returns the axis lengths in order.
    pub fn axes(&self) -> &[usize] {
        match &self.0 {
            DimensionStorage::Inline { axes, rank } => &axes[..rank.len()],
            DimensionStorage::Dynamic(axes) => axes,
        }
    }
}
impl From<&[usize]> for Dimensions {
    fn from(axes: &[usize]) -> Self {
        match axes {
            [] => Self::scalar(),
            [a] => Self::one(*a),
            [a, b] => Self::two(*a, *b),
            _ => Self::from(axes.to_vec()),
        }
    }
}
impl<const N: usize> From<&[usize; N]> for Dimensions {
    fn from(axes: &[usize; N]) -> Self {
        Self::from(axes.as_slice())
    }
}
impl From<usize> for Dimensions {
    fn from(n: usize) -> Self {
        Self::one(n)
    }
}
impl From<[usize; 1]> for Dimensions {
    fn from(v: [usize; 1]) -> Self {
        Self::one(v[0])
    }
}
impl From<[usize; 2]> for Dimensions {
    fn from(v: [usize; 2]) -> Self {
        Self::two(v[0], v[1])
    }
}
/// Converts axis lengths into a shape.
impl From<Vec<usize>> for Dimensions {
    fn from(axes: Vec<usize>) -> Self {
        match axes.as_slice() {
            [] => Self::scalar(),
            [a] => Self::one(*a),
            [a, b] => Self::two(*a, *b),
            _ => Self(DimensionStorage::Dynamic(axes)),
        }
    }
}
/// Converts axis lengths into a shape.
impl From<Box<[usize]>> for Dimensions {
    fn from(axes: Box<[usize]>) -> Self {
        Self::from(axes.into_vec())
    }
}
impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.axes()).finish()
    }
}

/// Mathematical site of a differentiation callback failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationSite {
    /// Unperturbed function validation.
    BasePoint,
    /// A Jacobian-vector product, including its base validation.
    Direction,
    /// One coordinate derivative.
    Variable(usize),
    /// Two-coordinate derivative (including Hessian diagonals).
    Variables(usize, usize),
}

/// Where a differentiation callback failed: the site and, if known, the output component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct DifferentiationLocation {
    site: DifferentiationSite,
    component: Option<usize>,
}
impl DifferentiationLocation {
    /// Returns the location of the callback at the unperturbed point.
    pub const fn base_point() -> Self {
        Self {
            site: DifferentiationSite::BasePoint,
            component: None,
        }
    }
    /// Returns the location of a directional derivative, including its base-point evaluation.
    pub const fn direction() -> Self {
        Self {
            site: DifferentiationSite::Direction,
            component: None,
        }
    }
    /// Returns the location of the derivative with respect to `variable`.
    pub const fn variable(variable: usize) -> Self {
        Self {
            site: DifferentiationSite::Variable(variable),
            component: None,
        }
    }
    /// Returns the location of the second derivative with respect to `first` and `second`.
    pub const fn variables(first: usize, second: usize) -> Self {
        Self {
            site: DifferentiationSite::Variables(first, second),
            component: None,
        }
    }
    /// Returns a copy with output component `component`.
    pub const fn with_component(mut self, component: usize) -> Self {
        self.component = Some(component);
        self
    }
    /// Returns the site.
    pub const fn site(&self) -> DifferentiationSite {
        self.site
    }
    /// Returns the output component, if known.
    pub const fn component_index(&self) -> Option<usize> {
        self.component
    }
}
impl fmt::Display for DifferentiationLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.site {
            DifferentiationSite::BasePoint => formatter.write_str("base point")?,
            DifferentiationSite::Direction => formatter.write_str("direction")?,
            DifferentiationSite::Variable(index) => write!(formatter, "variable {index}")?,
            DifferentiationSite::Variables(first, second) => {
                write!(formatter, "variable {first}, variable {second}")?
            }
        }
        if let Some(component) = self.component {
            write!(formatter, ", output component {component}")?;
        }
        Ok(())
    }
}

/// Structural reason an input or configuration is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum InvalidInputReason {
    /// A required collection or value is absent.
    Empty,
    /// A value lies outside an API or configuration range.
    OutOfRange,
    /// Individually valid inputs are inconsistent with each other.
    Inconsistent,
    /// Encoded or structured input is malformed.
    Malformed,
    /// The requested combination is not supported by this configuration.
    UnsupportedConfiguration,
}

impl fmt::Display for InvalidInputReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "empty input",
            Self::OutOfRange => "value outside the supported range",
            Self::Inconsistent => "inconsistent inputs",
            Self::Malformed => "malformed input",
            Self::UnsupportedConfiguration => "unsupported configuration",
        })
    }
}

/// Reason a finite mathematical argument is outside an operation's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum DomainReason {
    /// The argument is outside the function's mathematical domain.
    OutsideDomain,
    /// The argument is a pole of the requested function.
    Pole,
    /// The requested mathematical quantity is undefined.
    Undefined,
}

impl fmt::Display for DomainReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OutsideDomain => "outside the mathematical domain",
            Self::Pole => "function pole",
            Self::Undefined => "mathematically undefined",
        })
    }
}

/// Where a non-finite value was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum NonFiniteOrigin {
    /// Supplied directly to the public operation.
    Input,
    /// Returned by a user-provided callback, RNG, or analogous protocol provider.
    Callback,
    /// Produced inside the algorithm before its final result and not otherwise
    /// classified as finite-arithmetic overflow.
    Intermediate,
    /// Produced as the operation's final result and not otherwise classified as
    /// finite-arithmetic overflow.
    Output,
}

impl fmt::Display for NonFiniteOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Input => "input",
            Self::Callback => "callback result",
            Self::Intermediate => "intermediate value",
            Self::Output => "output",
        })
    }
}

/// Type of detected finite-precision numerical instability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum NumericalIssue {
    /// A normalized extended-range exponent cannot be represented by `i64`.
    /// This is distinct from overflow or underflow at an IEEE projection.
    ExponentOutOfRange,
    /// Finite operands exceeded the representable finite range, or overflow was
    /// predicted before returning an unusable value.
    Overflow,
    /// Underflow destroyed information required by the algorithm. Correctly
    /// rounded zero and subnormal results are not errors by themselves.
    Underflow,
    /// Catastrophic cancellation destroyed usable precision.
    Cancellation,
    /// The result is finite, but too many significant bits were lost for the
    /// requested accuracy contract.
    LossOfSignificance,
    /// Ill-conditioning prevents a trustworthy result.
    IllConditioned,
    /// A dense eigenproblem has no reliably computable complete basis of eigenvectors.
    DefectiveEigenbasis,
    /// An algorithmic denominator, factorization, or invariant broke down.
    Breakdown,
}

impl fmt::Display for NumericalIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ExponentOutOfRange => "extended exponent outside the supported range",
            Self::Overflow => "overflow",
            Self::Underflow => "underflow",
            Self::Cancellation => "catastrophic cancellation",
            Self::LossOfSignificance => "loss of significance",
            Self::IllConditioned => "ill-conditioned computation",
            Self::DefectiveEigenbasis => "defective or numerically defective eigenbasis",
            Self::Breakdown => "algorithmic breakdown",
        })
    }
}

/// Reason an otherwise valid finite iterative process did not converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum ConvergenceReason {
    /// The iteration budget was exhausted.
    IterationLimit,
    /// The evaluation budget was exhausted.
    EvaluationLimit,
    /// A non-iteration, non-evaluation solve resource was exhausted.
    ResourceLimit(crate::BudgetKind),
    /// The configured finite expansion could not be refined further.
    TruncationLimit,
    /// Progress stalled at finite values.
    Stagnation,
    /// A required step became smaller than the permitted floor.
    StepSizeLimit,
    /// A stored numerical trajectory could not be reproduced under its replay policy.
    ReplayMismatch,
    /// Finite iterates diverged from a solution.
    Divergence,
    /// A required nested solver did not converge.
    NestedSolver,
    /// Every independent candidate or restart failed.
    NoSuccessfulCandidate,
}

impl fmt::Display for ConvergenceReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::IterationLimit => "iteration limit reached",
            Self::EvaluationLimit => "evaluation limit reached",
            Self::ResourceLimit(_) => "resource limit reached",
            Self::TruncationLimit => "truncation limit reached",
            Self::Stagnation => "stagnation",
            Self::StepSizeLimit => "step-size limit reached",
            Self::ReplayMismatch => "trajectory replay mismatch",
            Self::Divergence => "divergence",
            Self::NestedSolver => "nested solver failure",
            Self::NoSuccessfulCandidate => "no successful candidate",
        })
    }
}

/// Reason a checked scalar conversion could not be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum ScalarConversionReason {
    /// The source value was NaN or infinite.
    NonFiniteInput,
    /// A finite source value lies outside the finite range of the target.
    OutOfRange,
    /// An exact conversion would change the value or the sign of zero.
    Inexact,
    /// The scalar implementation did not provide the requested conversion.
    Unsupported,
}

impl fmt::Display for ScalarConversionReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteInput => "the source value is not finite",
            Self::OutOfRange => "the source value is outside the target's finite range",
            Self::Inexact => "the conversion would lose precision",
            Self::Unsupported => "the scalar implementation does not support this conversion",
        })
    }
}

/// Machine-readable executor failure and recovery classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExecutionFailureReason {
    /// A worker thread could not be created; workers already started are joined.
    WorkerCreation {
        /// Original operating-system error code, if the platform supplied one.
        os_error: Option<i32>,
    },
    /// A scoped executor callback panicked.
    CallbackPanicked,
    /// An executor or callback returned an ordinary provider failure.
    ProviderFailure,
    /// The result's index, identity or range does not belong to the plan.
    UnknownPartition,
    /// A planned partition produced more than one result.
    DuplicatePartition,
    /// A planned partition has no result.
    MissingPartition,
    /// An injected executor violated another documented scheduling contract.
    ProtocolViolation,
}

impl fmt::Display for ExecutionFailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerCreation { os_error } => {
                f.write_str("worker creation")?;
                if let Some(code) = os_error {
                    write!(f, " (os error {code})")?;
                }
                Ok(())
            }
            Self::CallbackPanicked => f.write_str("callback panicked"),
            Self::ProviderFailure => f.write_str("provider failure"),
            Self::UnknownPartition => f.write_str("unknown partition"),
            Self::DuplicatePartition => f.write_str("duplicate partition"),
            Self::MissingPartition => f.write_str("missing partition"),
            Self::ProtocolViolation => f.write_str("protocol violation"),
        }
    }
}

/// Platform service or numerical operation required by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum Capability {
    /// A numerical operation required by the caller.
    Operation(&'static str),
    /// Caller-injected Basic Linear Algebra Subprograms provider.
    Blas,
    /// Monotonic clock and deadline observation.
    Clock,
    /// External entropy acquisition.
    Entropy,
    /// Host processor-feature observation.
    ProcessorFeatures,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Platform {
    Blas,
    Clock,
    Entropy,
    ProcessorFeatures,
}
impl Platform {
    const fn capability(self) -> Capability {
        match self {
            Self::Blas => Capability::Blas,
            Self::Clock => Capability::Clock,
            Self::Entropy => Capability::Entropy,
            Self::ProcessorFeatures => Capability::ProcessorFeatures,
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Operation(operation) => operation,
            Self::Blas => "blas provider",
            Self::Clock => "clock",
            Self::Entropy => "entropy",
            Self::ProcessorFeatures => "processor features",
        })
    }
}

/// Cooperative reason an execution stopped before completing all work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum ExecutionStopReason {
    /// Explicit cancellation was requested.
    Cancelled,
    /// The injected monotonic deadline was reached.
    DeadlineReached,
}

impl fmt::Display for ExecutionStopReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cancelled => "cancelled",
            Self::DeadlineReached => "deadline reached",
        })
    }
}

/// Stable top-level classification for [`SciOxideError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum SciOxideErrorKind {
    /// Structurally invalid input or configuration.
    InvalidInput,
    /// Incompatible dimensions.
    DimensionMismatch,
    /// Unrepresentable size or capacity.
    SizeOverflow,
    /// Corrupted accounting could not be reconciled exactly.
    AccountingInvariant,
    /// A configured resource limit was exceeded.
    ResourceLimit,
    /// Failed memory reservation.
    AllocationFailure,
    /// Injected executor construction, scheduling, or worker failure.
    ExecutionFailure,
    /// A requested platform or numerical capability was not supplied.
    CapabilityUnavailable,
    /// An injected platform capability failed.
    CapabilityFailure,
    /// Cooperative cancellation or deadline termination.
    ExecutionStopped,
    /// Invalid collection index.
    OutOfBounds,
    /// Finite mathematical domain violation.
    DomainError,
    /// Unexpected NaN or infinity.
    NonFinite,
    /// Failed checked scalar conversion.
    ScalarConversion,
    /// Valid finite process that did not converge.
    ConvergenceFailure,
    /// Singular matrix, optionally identifying a validated pivot.
    Singular,
    /// Finite-precision numerical breakdown.
    NumericalInstability,
    /// Failure at a differentiation stencil location.
    DifferentiationFailure,
    /// Deliberately unsupported operation.
    NotImplemented,
    /// Failure reported by a callback, provider, or application.
    External,
}

/// Error returned by SciOxide functions.
///
/// Classify it with [`Self::kind`] and read its fields with [`Self::view`]; see
/// [errors](crate#errors). Errors with a static description, shapes of rank at
/// most two and iteration-limit failures do not allocate. Other errors
/// allocate their description or diagnostics with the global allocator.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct SciOxideError(Repr);
#[derive(Debug, Clone, PartialEq)]
enum Repr {
    Inline(InlineCause),
    Boxed(Box<FullCause>),
}

// The shared error travels through every checked scalar `Result`.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<SciOxideError>() <= 40);
    assert!(size_of::<Result<f64, SciOxideError>>() <= 40);
    assert!(size_of::<Result<crate::Complex64, SciOxideError>>() <= 40);
};

/// Inline static diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
enum InlineCause {
    IterationLimit {
        own: usize,
        nested: usize,
        detail: &'static str,
    },
    InvalidInput {
        argument: Argument,
        reason: InvalidInputReason,
        detail: &'static str,
    },
    DimensionMismatch {
        axes: [[usize; 2]; 2],
        ranks: [InlineRank; 2],
    },
    SizeOverflow {
        context: Quantity,
    },
    AccountingInvariant {
        detail: &'static str,
        partition: Option<usize>,
    },
    ResourceLimit {
        resource: Quantity,
        required: usize,
        limit: usize,
    },
    AllocationFailure {
        context: Quantity,
        requested: usize,
    },
    ExecutionFailure {
        reason: ExecutionFailureReason,
        detail: &'static str,
    },
    CapabilityUnavailable {
        capability: Capability,
    },
    CapabilityFailure {
        capability: Platform,
        detail: &'static str,
    },
    ExecutionStopped {
        reason: ExecutionStopReason,
    },
    OutOfBounds {
        collection: Quantity,
        index: usize,
        size: usize,
    },
    DomainError {
        argument: Argument,
        reason: DomainReason,
        detail: &'static str,
    },
    NonFinite {
        origin: NonFiniteOrigin,
        detail: &'static str,
    },
    ScalarConversion {
        from_type: ConversionType,
        to_type: ConversionType,
        reason: ScalarConversionReason,
    },
    Singular {
        pivot: Option<usize>,
    },
    NumericalInstability {
        issue: NumericalIssue,
        detail: &'static str,
    },
    NotImplemented {
        detail: &'static str,
    },
    External {
        detail: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum FullCause {
    OperationFailure {
        operation: &'static str,
        detail: Cow<'static, str>,
    },
    Enriched {
        base: InlineCause,
        detail: Option<String>,
        partition: Option<usize>,
    },
    Shapes {
        expected: Dimensions,
        actual: Dimensions,
    },
    Convergence {
        reason: ConvergenceReason,
        diagnostics: Box<ConvergenceDiagnostics>,
        detail: Cow<'static, str>,
    },
    Differentiation {
        frames: Vec<DifferentiationFrame>,
        cause: Box<SciOxideError>,
    },
}
/// One enclosing differentiation call of a `DifferentiationFailure`; frames are listed outermost first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferentiationFrame {
    /// Mathematical location of the callback.
    pub location: DifferentiationLocation,
    /// Callback attempts, including the failing call.
    pub evaluations: usize,
}

fn split_detail(detail: Cow<'static, str>) -> (&'static str, Option<String>) {
    match detail {
        Cow::Borrowed(text) => (text, None),
        Cow::Owned(text) => ("", Some(text)),
    }
}

impl InlineCause {
    const fn kind(&self) -> SciOxideErrorKind {
        match self {
            Self::IterationLimit { .. } => SciOxideErrorKind::ConvergenceFailure,
            Self::InvalidInput { .. } => SciOxideErrorKind::InvalidInput,
            Self::DimensionMismatch { .. } => SciOxideErrorKind::DimensionMismatch,
            Self::SizeOverflow { .. } => SciOxideErrorKind::SizeOverflow,
            Self::AccountingInvariant { .. } => SciOxideErrorKind::AccountingInvariant,
            Self::ResourceLimit { .. } => SciOxideErrorKind::ResourceLimit,
            Self::AllocationFailure { .. } => SciOxideErrorKind::AllocationFailure,
            Self::ExecutionFailure { .. } => SciOxideErrorKind::ExecutionFailure,
            Self::CapabilityUnavailable { .. } => SciOxideErrorKind::CapabilityUnavailable,
            Self::CapabilityFailure { .. } => SciOxideErrorKind::CapabilityFailure,
            Self::ExecutionStopped { .. } => SciOxideErrorKind::ExecutionStopped,
            Self::OutOfBounds { .. } => SciOxideErrorKind::OutOfBounds,
            Self::DomainError { .. } => SciOxideErrorKind::DomainError,
            Self::NonFinite { .. } => SciOxideErrorKind::NonFinite,
            Self::ScalarConversion { .. } => SciOxideErrorKind::ScalarConversion,
            Self::Singular { .. } => SciOxideErrorKind::Singular,
            Self::NumericalInstability { .. } => SciOxideErrorKind::NumericalInstability,
            Self::NotImplemented { .. } => SciOxideErrorKind::NotImplemented,
            Self::External { .. } => SciOxideErrorKind::External,
        }
    }

    /// Project the cause, preferring boxed detail and context when present.
    fn view<'a>(
        &'a self,
        owned: Option<&'a str>,
        partition: Option<usize>,
    ) -> SciOxideErrorView<'a> {
        match *self {
            Self::IterationLimit {
                own,
                nested,
                detail,
            } => SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                diagnostics: ConvergenceDiagnosticsView::iterations(own, nested),
                detail,
            },
            Self::External { detail } => SciOxideErrorView::External {
                detail: owned.unwrap_or(detail),
            },
            Self::InvalidInput {
                argument,
                reason,
                detail,
                ..
            } => SciOxideErrorView::InvalidInput {
                argument,
                reason,
                detail: owned.unwrap_or(detail),
            },
            Self::DimensionMismatch {
                axes: [ref expected, ref actual],
                ranks: [expected_rank, actual_rank],
                ..
            } => {
                let (expected, actual) = (
                    &expected[..expected_rank.len()],
                    &actual[..actual_rank.len()],
                );
                SciOxideErrorView::DimensionMismatch { expected, actual }
            }
            Self::SizeOverflow { context, .. } => SciOxideErrorView::SizeOverflow { context },
            Self::AccountingInvariant {
                detail, partition, ..
            } => SciOxideErrorView::AccountingInvariant { detail, partition },
            Self::ResourceLimit {
                resource,
                required,
                limit,
                ..
            } => SciOxideErrorView::ResourceLimit {
                resource,
                required,
                limit,
            },
            Self::AllocationFailure {
                context, requested, ..
            } => SciOxideErrorView::AllocationFailure { context, requested },
            Self::ExecutionFailure { reason, detail, .. } => SciOxideErrorView::ExecutionFailure {
                partition,
                reason,
                detail: owned.unwrap_or(detail),
            },
            Self::CapabilityUnavailable { capability, .. } => {
                SciOxideErrorView::CapabilityUnavailable { capability }
            }
            Self::CapabilityFailure {
                capability, detail, ..
            } => SciOxideErrorView::CapabilityFailure {
                capability: capability.capability(),
                detail: owned.unwrap_or(detail),
            },
            Self::ExecutionStopped { reason, .. } => SciOxideErrorView::ExecutionStopped { reason },
            Self::OutOfBounds {
                collection,
                index,
                size,
                ..
            } => SciOxideErrorView::OutOfBounds {
                collection,
                index,
                size,
            },
            Self::DomainError {
                argument,
                reason,
                detail,
                ..
            } => SciOxideErrorView::DomainError {
                argument,
                reason,
                detail: owned.unwrap_or(detail),
            },
            Self::NonFinite { origin, detail, .. } => SciOxideErrorView::NonFinite {
                origin,
                detail: owned.unwrap_or(detail),
            },
            Self::ScalarConversion {
                from_type,
                to_type,
                reason,
                ..
            } => SciOxideErrorView::ScalarConversion {
                from_type,
                to_type,
                reason,
            },
            Self::Singular { pivot, .. } => SciOxideErrorView::Singular { pivot },
            Self::NumericalInstability { issue, detail, .. } => {
                SciOxideErrorView::NumericalInstability {
                    issue,
                    detail: owned.unwrap_or(detail),
                }
            }
            Self::NotImplemented { detail, .. } => SciOxideErrorView::NotImplemented {
                detail: owned.unwrap_or(detail),
            },
        }
    }
}

/// Logical borrowed classification independent of the owned representation.
#[derive(Debug)]
pub enum SciOxideErrorView<'a> {
    /// InvalidInput diagnostic.
    InvalidInput {
        /// Stable argument or configuration identity.
        argument: Argument,
        /// Machine-readable failure reason.
        reason: InvalidInputReason,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// DimensionMismatch diagnostic.
    DimensionMismatch {
        /// Expected axes.
        expected: &'a [usize],
        /// Observed axes.
        actual: &'a [usize],
    },
    /// SizeOverflow diagnostic.
    SizeOverflow {
        /// Checked resource or accounting boundary.
        context: Quantity,
    },
    /// AccountingInvariant diagnostic.
    AccountingInvariant {
        /// Explanation of the inconsistent accounting.
        detail: &'static str,
        /// Partition index, when the failure belongs to a partition.
        partition: Option<usize>,
    },
    /// Configured resource-limit diagnostic.
    ResourceLimit {
        /// Resource whose limit was exceeded.
        resource: Quantity,
        /// Required amount in the resource's documented units.
        required: usize,
        /// Configured maximum amount.
        limit: usize,
    },
    /// AllocationFailure diagnostic.
    AllocationFailure {
        /// Checked resource or accounting boundary.
        context: Quantity,
        /// Requested collection capacity.
        requested: usize,
    },
    /// ExecutionFailure diagnostic.
    ExecutionFailure {
        /// Partition index, when the failure belongs to a partition and was stored.
        partition: Option<usize>,
        /// Machine-readable failure reason.
        reason: ExecutionFailureReason,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// CapabilityUnavailable diagnostic.
    CapabilityUnavailable {
        /// Requested platform capability.
        capability: Capability,
    },
    /// CapabilityFailure diagnostic.
    CapabilityFailure {
        /// Requested platform capability.
        capability: Capability,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// ExecutionStopped diagnostic.
    ExecutionStopped {
        /// Machine-readable failure reason.
        reason: ExecutionStopReason,
    },
    /// OutOfBounds diagnostic.
    OutOfBounds {
        /// Collection whose index was checked.
        collection: Quantity,
        /// Zero-based checked index.
        index: usize,
        /// Collection length.
        size: usize,
    },
    /// DomainError diagnostic.
    DomainError {
        /// Stable argument or configuration identity.
        argument: Argument,
        /// Machine-readable failure reason.
        reason: DomainReason,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// NonFinite diagnostic.
    NonFinite {
        /// Origin of the non-finite value.
        origin: NonFiniteOrigin,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// ScalarConversion diagnostic.
    ScalarConversion {
        /// Source type.
        from_type: ConversionType,
        /// Destination type.
        to_type: ConversionType,
        /// Machine-readable failure reason.
        reason: ScalarConversionReason,
    },
    /// ConvergenceFailure diagnostic.
    ConvergenceFailure {
        /// Machine-readable failure reason.
        reason: ConvergenceReason,
        /// Complete diagnostic levels.
        diagnostics: ConvergenceDiagnosticsView<'a>,

        /// Human-readable detail.
        detail: &'a str,
    },
    /// Singular diagnostic.
    Singular {
        /// Validated zero-based pivot information, when available.
        pivot: Option<usize>,
    },
    /// NumericalInstability diagnostic.
    NumericalInstability {
        /// Machine-readable numerical breakdown.
        issue: NumericalIssue,
        /// Human-readable detail.
        detail: &'a str,
    },
    /// Callback failure inside numerical differentiation, with every enclosing differentiation call.
    DifferentiationFailure {
        /// Boundary frames, outermost first.
        frames: &'a [DifferentiationFrame],
        /// Original non-differentiation cause.
        cause: &'a SciOxideError,
    },
    /// NotImplemented diagnostic.
    NotImplemented {
        /// Human-readable detail.
        detail: &'a str,
    },
    /// Failure from code outside SciOxide.
    External {
        /// Human-readable explanation supplied by the caller's code.
        detail: &'a str,
    },
}
impl SciOxideError {
    /// Returns the kind-specific fields of the error.
    pub fn view(&self) -> SciOxideErrorView<'_> {
        match &self.0 {
            Repr::Inline(base) => base.view(None, None),
            Repr::Boxed(full) => match full.as_ref() {
                FullCause::OperationFailure { operation, detail } => {
                    SciOxideErrorView::CapabilityFailure {
                        capability: Capability::Operation(operation),
                        detail,
                    }
                }
                FullCause::Enriched {
                    base,
                    detail,
                    partition,
                } => base.view(detail.as_deref(), *partition),
                FullCause::Shapes { expected, actual } => SciOxideErrorView::DimensionMismatch {
                    expected: expected.axes(),
                    actual: actual.axes(),
                },
                FullCause::Convergence {
                    reason,
                    diagnostics,
                    detail,
                } => SciOxideErrorView::ConvergenceFailure {
                    reason: *reason,
                    diagnostics: ConvergenceDiagnosticsView::full(diagnostics),
                    detail,
                },
                FullCause::Differentiation { frames, cause } => {
                    SciOxideErrorView::DifferentiationFailure { frames, cause }
                }
            },
        }
    }
}
struct Axes<'a>(&'a [usize]);
impl fmt::Display for Axes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0).finish()
    }
}

fn display_error(view: SciOxideErrorView<'_>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match view {
        SciOxideErrorView::External { detail } => write!(formatter, "external failure: {detail}"),
        SciOxideErrorView::InvalidInput {
            argument,
            reason,
            detail,
        } => write!(formatter, "invalid input `{argument}` ({reason}): {detail}"),
        SciOxideErrorView::DimensionMismatch { expected, actual } => write!(
            formatter,
            "dimension mismatch: expected {}, got {}",
            Axes(expected),
            Axes(actual)
        ),
        SciOxideErrorView::AccountingInvariant { detail, partition } => {
            write!(formatter, "accounting invariant failure: {detail}")?;
            if let Some(index) = partition {
                write!(formatter, " (partition {index})")?;
            }
            Ok(())
        }
        SciOxideErrorView::SizeOverflow { context } => {
            write!(formatter, "size overflow while computing {context}")
        }
        SciOxideErrorView::ResourceLimit {
            resource,
            required,
            limit,
        } => write!(
            formatter,
            "resource limit: {resource} requires {required}, limit {limit}"
        ),
        SciOxideErrorView::AllocationFailure { context, requested } => write!(
            formatter,
            "allocation failed for {context} (requested capacity: {requested})"
        ),
        SciOxideErrorView::ExecutionFailure {
            partition,
            reason,
            detail,
        } => match partition {
            Some(partition) => write!(
                formatter,
                "execution failed for partition {partition} ({reason}): {detail}"
            ),
            None => write!(formatter, "execution failed ({reason}): {detail}"),
        },
        SciOxideErrorView::CapabilityUnavailable { capability } => {
            write!(formatter, "capability {capability} is unavailable")
        }
        SciOxideErrorView::CapabilityFailure { capability, detail } => {
            write!(formatter, "capability {capability} failed: {detail}")
        }
        SciOxideErrorView::ExecutionStopped { reason } => {
            write!(formatter, "execution stopped: {reason}")
        }
        SciOxideErrorView::OutOfBounds {
            collection,
            index,
            size,
        } => write!(
            formatter,
            "out of bounds in {collection}: index {index} for size {size}"
        ),
        SciOxideErrorView::DomainError {
            argument,
            reason,
            detail,
        } => write!(
            formatter,
            "domain error for `{argument}` ({reason}): {detail}"
        ),
        SciOxideErrorView::NonFinite { origin, detail } => {
            write!(formatter, "non-finite {origin}: {detail}")
        }
        SciOxideErrorView::ScalarConversion {
            from_type,
            to_type,
            reason,
        } => write!(
            formatter,
            "scalar conversion from {from_type} to {to_type} failed: {reason}"
        ),
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            detail,
            ..
        } => {
            write!(formatter, "convergence failure ({reason}): {detail}")?;
            diagnostics.fmt(formatter)
        }
        SciOxideErrorView::Singular { pivot } => {
            write!(formatter, "singular matrix")?;
            if let Some(pivot) = pivot {
                write!(formatter, " at pivot {}", pivot)?;
            }
            Ok(())
        }
        SciOxideErrorView::NumericalInstability { issue, detail } => {
            write!(formatter, "numerical instability ({issue}): {detail}")
        }
        SciOxideErrorView::DifferentiationFailure { frames, .. } => {
            for (index, frame) in frames.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(": ")?;
                }
                write!(
                    formatter,
                    "differentiation failure at {} after {} callback evaluations",
                    frame.location, frame.evaluations
                )?;
            }
            Ok(())
        }
        SciOxideErrorView::NotImplemented { detail } => {
            write!(formatter, "not implemented: {detail}")
        }
    }
}
impl fmt::Display for SciOxideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_error(self.view(), f)
    }
}
impl core::error::Error for SciOxideError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self.view() {
            SciOxideErrorView::DifferentiationFailure { cause, .. } => Some(cause),
            _ => None,
        }
    }
}

impl SciOxideError {
    /// Returns the error kind.
    pub fn kind(&self) -> SciOxideErrorKind {
        match &self.0 {
            Repr::Inline(cause) => cause.kind(),
            Repr::Boxed(full) => match full.as_ref() {
                FullCause::OperationFailure { .. } => SciOxideErrorKind::CapabilityFailure,
                FullCause::Enriched { base, .. } => base.kind(),
                FullCause::Differentiation { .. } => SciOxideErrorKind::DifferentiationFailure,
                FullCause::Shapes { .. } => SciOxideErrorKind::DimensionMismatch,
                FullCause::Convergence { .. } => SciOxideErrorKind::ConvergenceFailure,
            },
        }
    }

    const fn inline(cause: InlineCause) -> Self {
        Self(Repr::Inline(cause))
    }

    fn assemble(base: InlineCause, detail: Option<String>, partition: Option<usize>) -> Self {
        if detail.is_none() && partition.is_none() {
            Self::inline(base)
        } else {
            Self(Repr::Boxed(Box::new(FullCause::Enriched {
                base,
                detail,
                partition,
            })))
        }
    }

    /// Creates an `External` error for a failure in your own callback or application code.
    ///
    /// SciOxide returns such errors unchanged. A `&'static str` description does
    /// not allocate.
    ///
    /// ```
    /// use scioxide_core::{SciOxideError, SciOxideErrorKind};
    ///
    /// fn model(x: f64) -> Result<f64, SciOxideError> {
    ///     if x <= 0.0 {
    ///         return Err(SciOxideError::external("model requires x > 0"));
    ///     }
    ///     Ok(x.ln())
    /// }
    /// assert_eq!(model(0.0).unwrap_err().kind(), SciOxideErrorKind::External);
    /// ```
    pub fn external(detail: impl Into<Cow<'static, str>>) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(InlineCause::External { detail }, owned, None)
    }

    /// Creates an `InvalidInput` error with a static description; usable in `const` context.
    pub const fn invalid_input_static(
        argument: Argument,
        reason: InvalidInputReason,
        detail: &'static str,
    ) -> Self {
        Self::inline(InlineCause::InvalidInput {
            argument,
            reason,
            detail,
        })
    }
    /// Creates an `InvalidInput` error for `argument`.
    #[cold]
    #[inline(never)]
    pub fn invalid_input(
        argument: Argument,
        reason: InvalidInputReason,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(
            InlineCause::InvalidInput {
                argument,
                reason,
                detail,
            },
            owned,
            None,
        )
    }
    /// Creates a `DimensionMismatch` error from the expected and actual shapes.
    pub fn dimension_mismatch(
        expected: impl Into<Dimensions>,
        actual: impl Into<Dimensions>,
    ) -> Self {
        let (expected, actual) = (expected.into(), actual.into());
        match (&expected.0, &actual.0) {
            (
                DimensionStorage::Inline { axes: e, rank: er },
                DimensionStorage::Inline { axes: a, rank: ar },
            ) => Self::inline(InlineCause::DimensionMismatch {
                axes: [*e, *a],
                ranks: [*er, *ar],
            }),
            _ => Self(Repr::Boxed(Box::new(FullCause::Shapes {
                expected,
                actual,
            }))),
        }
    }
    /// Creates a `SizeOverflow` error for `context`.
    #[cold]
    #[inline(never)]
    pub const fn size_overflow(context: Quantity) -> Self {
        Self::inline(InlineCause::SizeOverflow { context })
    }
    /// Creates a `ResourceLimit` error: `resource` needs `required` units but the limit is `limit`.
    ///
    /// The units are those documented for `resource`.
    ///
    /// ```
    /// use scioxide_core::{SciOxideError, SciOxideErrorKind, SciOxideErrorView, quantity};
    ///
    /// let error = SciOxideError::resource_limit(quantity!("workspace_bytes"), 128, 64);
    /// assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    /// assert!(matches!(
    ///     error.view(),
    ///     SciOxideErrorView::ResourceLimit { required: 128, .. }
    /// ));
    /// ```
    pub const fn resource_limit(resource: Quantity, required: usize, limit: usize) -> Self {
        Self::inline(InlineCause::ResourceLimit {
            resource,
            required,
            limit,
        })
    }

    /// Creates an `AccountingInvariant` error, optionally for one partition.
    #[cold]
    #[inline(never)]
    pub const fn accounting_invariant(detail: &'static str, partition: Option<usize>) -> Self {
        Self::inline(InlineCause::AccountingInvariant { detail, partition })
    }
    /// Creates an `AllocationFailure` error for reserving capacity `requested` for `context`.
    pub const fn allocation_failure(context: Quantity, requested: usize) -> Self {
        Self::inline(InlineCause::AllocationFailure { context, requested })
    }
    /// Creates an `ExecutionFailure` error, optionally for one partition.
    pub fn execution_failure(
        partition: Option<usize>,
        reason: ExecutionFailureReason,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(
            InlineCause::ExecutionFailure { reason, detail },
            owned,
            partition,
        )
    }
    /// Creates a `CapabilityUnavailable` error.
    pub const fn capability_unavailable(capability: Capability) -> Self {
        Self::inline(InlineCause::CapabilityUnavailable { capability })
    }
    /// Creates a `CapabilityFailure` error.
    pub fn capability_failure(
        capability: Capability,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let capability = match capability {
            Capability::Operation(operation) => {
                return Self(Repr::Boxed(Box::new(FullCause::OperationFailure {
                    operation,
                    detail: detail.into(),
                })));
            }
            Capability::Blas => Platform::Blas,
            Capability::Clock => Platform::Clock,
            Capability::Entropy => Platform::Entropy,
            Capability::ProcessorFeatures => Platform::ProcessorFeatures,
        };
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(
            InlineCause::CapabilityFailure { capability, detail },
            owned,
            None,
        )
    }
    /// Creates an `ExecutionStopped` error.
    pub const fn execution_stopped(reason: ExecutionStopReason) -> Self {
        Self::inline(InlineCause::ExecutionStopped { reason })
    }
    /// Creates an `OutOfBounds` error: `index` is not below `size` for `collection`.
    pub const fn out_of_bounds(collection: Quantity, index: usize, size: usize) -> Self {
        Self::inline(InlineCause::OutOfBounds {
            collection,
            index,
            size,
        })
    }
    /// Creates a `DomainError` error for `argument`.
    pub fn domain_error(
        argument: Argument,
        reason: DomainReason,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(
            InlineCause::DomainError {
                argument,
                reason,
                detail,
            },
            owned,
            None,
        )
    }
    /// Creates a `NonFinite` error.
    #[cold]
    #[inline(never)]
    pub fn non_finite(origin: NonFiniteOrigin, detail: impl Into<Cow<'static, str>>) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(InlineCause::NonFinite { origin, detail }, owned, None)
    }
    /// Creates a `ScalarConversion` error.
    pub const fn scalar_conversion(
        from_type: ConversionType,
        to_type: ConversionType,
        reason: ScalarConversionReason,
    ) -> Self {
        Self::inline(InlineCause::ScalarConversion {
            from_type,
            to_type,
            reason,
        })
    }
    /// Creates a `ConvergenceFailure` error with its diagnostics.
    ///
    /// An iteration-limit failure with a static description and only iteration
    /// counts does not allocate. Other diagnostics are stored in full.
    pub fn convergence_failure(
        reason: ConvergenceReason,
        diagnostics: ConvergenceDiagnostics,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let detail = detail.into();
        if reason == ConvergenceReason::IterationLimit
            && let Cow::Borrowed(detail) = detail
            && let Some((own, nested)) = diagnostics.iteration_counts()
        {
            return Self::inline(InlineCause::IterationLimit {
                own,
                nested,
                detail,
            });
        }
        Self(Repr::Boxed(Box::new(FullCause::Convergence {
            reason,
            diagnostics: Box::new(diagnostics),
            detail,
        })))
    }
    /// Creates a `Singular` error, with the zero-based pivot index if known.
    pub const fn singular(pivot: Option<usize>) -> Self {
        Self::inline(InlineCause::Singular { pivot })
    }
    /// Creates a `NumericalInstability` error.
    #[cold]
    #[inline(never)]
    pub fn numerical_instability(
        issue: NumericalIssue,
        detail: impl Into<Cow<'static, str>>,
    ) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(
            InlineCause::NumericalInstability { issue, detail },
            owned,
            None,
        )
    }
    /// Creates a `NotImplemented` error.
    pub fn not_implemented(detail: impl Into<Cow<'static, str>>) -> Self {
        let (detail, owned) = split_detail(detail.into());
        Self::assemble(InlineCause::NotImplemented { detail }, owned, None)
    }
    /// Wraps `source` as a `DifferentiationFailure` at `location`, after `evaluations` callback attempts.
    ///
    /// Wrapping an existing differentiation failure adds the call as the new
    /// outermost frame. `source()` returns the original cause.
    pub fn differentiation_failure(
        location: DifferentiationLocation,
        evaluations: usize,
        source: Self,
    ) -> Self {
        let frame = DifferentiationFrame {
            location,
            evaluations,
        };
        let mut source = source;
        if let Repr::Boxed(full) = &mut source.0
            && let FullCause::Differentiation { frames, .. } = full.as_mut()
        {
            frames.insert(0, frame);
            return source;
        }
        Self(Repr::Boxed(Box::new(FullCause::Differentiation {
            frames: alloc::vec![frame],
            cause: Box::new(source),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use core::fmt::Write;

    struct Sink(usize);
    impl Write for Sink {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
            Ok(())
        }
    }
    #[test]
    fn compact_iteration_counts_preserve_overflowing_totals() {
        let counts = crate::SolveCounts::from_iteration_counts(usize::MAX, 1);
        let error = SciOxideError::convergence_failure(
            ConvergenceReason::IterationLimit,
            ConvergenceDiagnostics::new().with_counts(counts),
            "work exhausted",
        );
        let SciOxideErrorView::ConvergenceFailure { diagnostics, .. } = error.view() else {
            unreachable!()
        };
        assert_eq!(diagnostics.counts(), counts);
        assert!(diagnostics.counts().total().is_err());
    }

    #[test]
    fn deeply_nested_diagnostics_format_with_constant_stack() {
        let identity = crate::algorithm_id!("test", "nested", 1);
        let mut diagnostics = ConvergenceDiagnostics::new();
        let mut error = SciOxideError::singular(None);
        for index in 0..4096 {
            diagnostics = ConvergenceDiagnostics::new().with_nested(identity, diagnostics);
            error = SciOxideError::differentiation_failure(
                DifferentiationLocation::variable(index),
                index,
                error,
            );
        }
        let copy = diagnostics.clone();
        assert_eq!(copy, diagnostics);
        assert_eq!(copy.nested_levels().len(), 4096);
        let mut sink = Sink(0);
        write!(sink, "{copy}").unwrap();
        assert_eq!(sink.0, 4096 * " [nested=test/nested@1]".len());
        write!(sink, "{copy:?}").unwrap();
        let copied_error = error.clone();
        assert_eq!(copied_error, error);
        write!(sink, "{copied_error} {copied_error:?}").unwrap();
        let SciOxideErrorView::DifferentiationFailure { frames, cause } = error.view() else {
            unreachable!()
        };
        assert_eq!(frames.len(), 4096);
        assert_eq!(frames[0].evaluations, 4095);
        assert_eq!(frames[4095].evaluations, 0);
        assert_eq!(cause.kind(), SciOxideErrorKind::Singular);
        drop((copy, diagnostics, copied_error, error));
    }
    #[test]
    fn dimensions_keep_axes_and_inline_small_ranks() {
        for dimensions in [
            Dimensions::scalar(),
            Dimensions::one(3),
            Dimensions::two(3, 4),
        ] {
            assert!(matches!(dimensions.0, DimensionStorage::Inline { .. }));
        }
        let tensor = Dimensions::from(&[2, 3, 4]);
        assert!(matches!(tensor.0, DimensionStorage::Dynamic(_)));
        assert_eq!(tensor.axes(), &[2, 3, 4]);
        assert_eq!(tensor.rank(), 3);
    }
    #[test]
    fn owned_dimensions_preserve_axes_and_storage_policy() {
        for axes in [
            alloc::vec![],
            alloc::vec![2],
            alloc::vec![2, 3],
            alloc::vec![2, 3, 4],
        ] {
            let vector = Dimensions::from(axes.clone());
            let boxed = Dimensions::from(axes.clone().into_boxed_slice());
            for dimensions in [vector, boxed] {
                assert_eq!(dimensions.axes(), axes);
                assert_eq!(dimensions.rank(), axes.len());
                assert_eq!(
                    matches!(dimensions.0, DimensionStorage::Inline { .. }),
                    axes.len() <= 2,
                );
            }
        }
    }
    #[test]
    fn display_remains_human_readable() {
        let error = SciOxideError::invalid_input(
            crate::argument!("size"),
            InvalidInputReason::OutOfRange,
            "must be positive",
        );
        assert_eq!(
            error.to_string(),
            "invalid input `size` (value outside the supported range): must be positive"
        );
        let error = SciOxideError::dimension_mismatch([2, 3], [4, 5]);
        assert_eq!(
            error.to_string(),
            "dimension mismatch: expected [2, 3], got [4, 5]"
        );
        let plain = SciOxideError::execution_failure(
            None,
            ExecutionFailureReason::UnknownPartition,
            "unknown",
        )
        .to_string();
        let partitioned = SciOxideError::execution_failure(
            Some(2),
            ExecutionFailureReason::UnknownPartition,
            "unknown",
        )
        .to_string();
        assert_eq!(partitioned.replace(" for partition 2", ""), plain);
    }

    #[test]
    fn every_view_has_identity_free_display() {
        let cases = [
            (
                SciOxideError::invalid_input(
                    crate::argument!("x"),
                    InvalidInputReason::Empty,
                    "missing",
                ),
                "invalid input `x` (empty input): missing",
            ),
            (
                SciOxideError::dimension_mismatch([2, 3], [4, 5]),
                "dimension mismatch: expected [2, 3], got [4, 5]",
            ),
            (
                SciOxideError::accounting_invariant("counts", None),
                "accounting invariant failure: counts",
            ),
            (
                SciOxideError::accounting_invariant("counts", Some(2)),
                "accounting invariant failure: counts (partition 2)",
            ),
            (
                SciOxideError::size_overflow(crate::quantity!("size")),
                "size overflow while computing size",
            ),
            (
                SciOxideError::resource_limit(crate::quantity!("storage"), 7, 4),
                "resource limit: storage requires 7, limit 4",
            ),
            (
                SciOxideError::allocation_failure(crate::quantity!("storage"), 7),
                "allocation failed for storage (requested capacity: 7)",
            ),
            (
                SciOxideError::execution_failure(
                    Some(2),
                    ExecutionFailureReason::ProviderFailure,
                    "failed",
                ),
                "execution failed for partition 2 (provider failure): failed",
            ),
            (
                SciOxideError::execution_failure(
                    None,
                    ExecutionFailureReason::ProviderFailure,
                    "failed",
                ),
                "execution failed (provider failure): failed",
            ),
            (
                SciOxideError::capability_unavailable(Capability::Clock),
                "capability clock is unavailable",
            ),
            (
                SciOxideError::capability_failure(Capability::Clock, "failed"),
                "capability clock failed: failed",
            ),
            (
                SciOxideError::execution_stopped(ExecutionStopReason::Cancelled),
                "execution stopped: cancelled",
            ),
            (
                SciOxideError::out_of_bounds(crate::quantity!("values"), 3, 2),
                "out of bounds in values: index 3 for size 2",
            ),
            (
                SciOxideError::domain_error(crate::argument!("x"), DomainReason::Pole, "pole"),
                "domain error for `x` (function pole): pole",
            ),
            (
                SciOxideError::non_finite(NonFiniteOrigin::Callback, "NaN"),
                "non-finite callback result: NaN",
            ),
            (
                SciOxideError::scalar_conversion(
                    ConversionType::F64,
                    ConversionType::F32,
                    ScalarConversionReason::OutOfRange,
                ),
                "scalar conversion from f64 to f32 failed: the source value is outside the target's finite range",
            ),
            (
                SciOxideError::convergence_failure(
                    ConvergenceReason::IterationLimit,
                    ConvergenceDiagnostics::new(),
                    "limit",
                ),
                "convergence failure (iteration limit reached): limit",
            ),
            (SciOxideError::singular(None), "singular matrix"),
            (
                SciOxideError::singular(Some(2)),
                "singular matrix at pivot 2",
            ),
            (
                SciOxideError::numerical_instability(NumericalIssue::Overflow, "large"),
                "numerical instability (overflow): large",
            ),
            (
                SciOxideError::not_implemented("unavailable"),
                "not implemented: unavailable",
            ),
            (
                SciOxideError::external("model failed"),
                "external failure: model failed",
            ),
            (
                SciOxideError::differentiation_failure(
                    DifferentiationLocation::variable(1),
                    2,
                    SciOxideError::differentiation_failure(
                        DifferentiationLocation::base_point(),
                        1,
                        SciOxideError::external("model failed"),
                    ),
                ),
                "differentiation failure at variable 1 after 2 callback evaluations: differentiation failure at base point after 1 callback evaluations",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn malformed_identifier_conversion_is_const_and_typed() {
        const ERROR: SciOxideError = match crate::Argument::new("bad path") {
            Err(error) => error.into_scioxide(),
            Ok(_) => unreachable!(),
        };
        assert!(matches!(ERROR.view(), SciOxideErrorView::InvalidInput {
            argument, reason: InvalidInputReason::Malformed, ..
        } if argument == crate::argument!("identifier")));
    }
}
