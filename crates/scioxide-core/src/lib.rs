//! Scalar types, checked arithmetic, errors and execution shared by all SciOxide crates.
//!
//! `scioxide-core` defines what every SciOxide crate builds on: the scalar traits
//! [`RealScalar`] and [`NumericScalar`], [`Complex`] numbers, the error type
//! [`SciOxideError`], tolerances, intervals and budgets, borrowed strided arrays,
//! and the execution and solve-accounting types that solvers take as arguments.
//! It also contains numerical functions that are useful on their own: correctly
//! rounded sums and dot products, compensated summation, log-sum-exp and
//! softmax without overflow, `sin(πx)` and related functions, values with an
//! extended exponent range, and exact combinatorial counts.
//!
//! The sections from [Errors](#errors) to [Feature flags](#feature-flags) state
//! rules that hold in every SciOxide crate. Other crates link to them instead of
//! repeating them.
//!
//! # Quick start
//!
//! ```
//! use scioxide_core::{SciOxideErrorKind, checked_sum, logsumexp};
//!
//! // The exact sum is 1; naive left-to-right summation returns 0.
//! assert_eq!(checked_sum(&[1.0e100_f64, 1.0, -1.0e100])?, 1.0);
//!
//! // log(e¹⁰⁰⁰ + e¹⁰⁰⁰) = 1000 + ln 2, although e¹⁰⁰⁰ overflows `f64`.
//! let log = logsumexp(&[1000.0_f64, 1000.0])?;
//! assert!((log - (1000.0 + core::f64::consts::LN_2)).abs() < 1e-12);
//!
//! let error = checked_sum(&[1.0_f64, f64::NAN]).unwrap_err();
//! assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Task | Entry points | Method |
//! |---|---|---|
//! | Sum, dot product, midpoint rounded once | [`checked_sum`], [`checked_sum_of_products`], [`checked_midpoint`] | exact accumulation, one final rounding |
//! | Running sums in working precision | [`CompensatedSum`], [`compensated_reduce`] | Neumaier-style compensated summation |
//! | Sums and products beyond the `f64` exponent range | [`ScaledAccumulator`], [`Scaled`], [`PromotingCompensatedSum`] | separate mantissa and integer exponent |
//! | Euclidean norm without overflow | [`ScaledNorm`], [`checked_scaled_norm`] | Scaled sum of squares |
//! | Log of a sum of exponentials | [`logsumexp`], [`weighted_logsumexp`], [`logsumexp_axis`] | |
//! | Softmax along an axis | [`softmax`], [`log_softmax`], [`softmax_axis`], [`log_softmax_axis`] | |
//! | Trigonometry in units of π or degrees | [`sin_pi`], [`cos_pi`], [`sincpi`], [`sin_degrees`] | |
//! | Functions that cancel near a point | [`cosm1`], [`exprel`], [`powm1`], [`log1mexp`], [`xlogy`], [`xlog1py`] | |
//! | Logistic functions | [`sigmoid`], [`log_sigmoid`], [`logit`], [`softplus`] | |
//! | Exact counts | [`checked_binomial`], [`checked_factorial`], [`checked_stirling_second_kind`] | integer arithmetic |
//! | Combinations and permutations | [`CombinationEnumerator`], [`Permutation`], [`permutation_rank`] | lexicographic order |
//! | Intervals and variable bounds | [`Interval`], [`ExtendedInterval`], [`Bounds`] | |
//! | Convergence tolerances | [`Tolerance`], [`StepTolerance`], [`ResidualTolerance`] | |
//! | Strided multidimensional arrays | [`StridedLayout`], [`NdView`], [`NdViewMut`], [`StridedLayout::row_major`], [`NdView::row_major`] | |
//! | Matrix-free linear operators | [`LinearOperator`], [`Preconditioner`], [`ClosureOperator`], [`ClosurePreconditioner`] | |
//!
//! # Accuracy
//!
//! `ε` is the machine epsilon of the scalar type and an ulp is the spacing of
//! floating-point numbers at the exact result.
//!
//! | Functions | Error |
//! |---|---|
//! | [`checked_sum`], [`checked_sum_of_products`], [`checked_difference_of_products`], [`checked_float_product`], [`checked_float_quotient`], [`checked_midpoint`], [`checked_affine`] | correctly rounded: the exact result rounded once |
//! | [`checked_product_quotient`] | within 4 ulps of the correctly rounded result |
//! | [`CompensatedSum`], [`compensated_reduce`] | working precision with a first-order correction; not correctly rounded |
//! | [`ScaledNorm`], [`checked_scaled_norm`] | relative error `O(n ε)`; the bound is stated on [`ScaledNorm`] |
//! | [`checked_quadratic_roots`] | small scaled residual; no forward error bound, since roots can be ill-conditioned |
//! | [`sin_pi`], [`cos_pi`], [`tan_pi`], [`sin_degrees`], [`cosm1`], [`sinc`], [`exprel`], [`powm1`], [`xlogy`], [`xlog1py`], [`sigmoid`], [`log_sigmoid`], [`logit`], [`softplus`] for real arguments | within 16 ulps; complex bounds are stated on each function |
//! | [`logsumexp`] of real values | within 16 ulps |
//!
//! These bounds hold for both `f32` and `f64`.
//!
//! # Errors
//!
//! Every fallible SciOxide function returns `Result<_, SciOxideError>`, except where
//! a crate defines its own error type. Classify an error with
//! [`SciOxideError::kind`], and read its fields with [`SciOxideError::view`]. Do not
//! match on its `Display` text.
//!
//! An error describes what failed. Where it applies, it names the rejected
//! parameter as an [`Argument`] (for example `options.max_steps`) or the size or
//! counter that failed as a [`Quantity`] (for example `workspace.capacity`).
//! It does not record which function reported it. Errors returned by your
//! callbacks propagate unchanged; code outside SciOxide creates one with
//! [`SciOxideError::external`].
//!
//! | Situation | [`SciOxideErrorKind`] |
//! |---|---|
//! | Argument outside its accepted range, empty input, inconsistent arguments | `InvalidInput`, with an [`InvalidInputReason`] |
//! | Finite argument outside the mathematical domain (for example `log` of a negative number) | `DomainError` |
//! | NaN or infinity where a finite value is required | `NonFinite`, with a [`NonFiniteOrigin`] |
//! | Overflow from finite operands, or underflow that loses the result | `NumericalInstability`, with a [`NumericalIssue`] |
//! | Shapes that do not agree, or a buffer that is too short | `DimensionMismatch` |
//! | Size arithmetic that overflows `usize` | `SizeOverflow` |
//! | Configured memory or output limit exceeded | `ResourceLimit` |
//! | Memory reservation failed | `AllocationFailure` |
//! | Iteration did not converge within its budget | `ConvergenceFailure` |
//! | Cancellation or deadline | `ExecutionStopped` |
//! | Missing capability, such as a clock for a time limit | `CapabilityUnavailable` |
//! | Singular matrix | `Singular` |
//!
//! Most errors do not allocate. `SciOxideError` is at most 40 bytes on 64-bit
//! targets, so `Result<f64, SciOxideError>` stays small.
//!
//! # Non-finite values
//!
//! Checked SciOxide functions reject NaN and infinite inputs with a `NonFinite` error whose
//! origin is [`NonFiniteOrigin::Input`]. A NaN or infinity returned by a callback
//! has origin [`NonFiniteOrigin::Callback`]. Exceptions are documented by the
//! owning API: logarithmic functions can accept `−∞` as a zero weight, endpoint
//! limits and condition diagnostics can be infinite, and automatic differentiation
//! follows its documented non-finite policy. A checked numerical value is otherwise
//! finite; for example, [`logsumexp`] of an empty slice is the limit `−∞`.
//! An infinity produced from finite operands is reported as overflow (`NumericalInstability`), not as `NonFinite`.
//!
//! Subnormal inputs and results are kept; nothing is flushed to zero. The sign of
//! zero is kept wherever it is observable. Empty sums and exact cancellations
//! return `+0`, and empty products return `1`.
//!
//! The arithmetic operators of the scalar traits follow IEEE 754 and do not check
//! anything. The checked functions in this crate do.
//!
//! # Output on error
//!
//! Functions validate all inputs before they write to an output. A function that
//! writes into a buffer you provide (usually named `*_into`) either writes the
//! complete result or, on error, leaves the output unchanged. Scratch buffers may
//! change on error but stay usable. Callbacks and random number generators may
//! have advanced before the error.
//!
//! Empty containers and reductions are valid when their operation has an identity,
//! such as `+0` for an empty sum; they invoke no element callbacks.
//! A rank-zero shape contains one scalar, while any zero extent makes its
//! cardinality zero, even if multiplying the other extents would overflow.
//! Solvers that require unknowns or observations reject empty problems.
//! Zero-dimensional integration instead evaluates its integrand once: the empty
//! product measure is one. Each operation documents which case applies.
//!
//! # Plans and workspaces
//!
//! A plan is an immutable object whose construction computes data that every
//! execution reuses, such as tables, nodes and weights, factorizations,
//! symbolic structure, search trees or schedules, so you can apply it to many
//! inputs. A value that only holds validated settings is an options or method
//! value, not a plan. A workspace is the scratch storage a plan needs. Plans
//! report their needs with `workspace_requirements`, and growable workspaces
//! offer `check`, which returns a [`Capacity`], and `try_reserve`. A workspace
//! whose borrowed scratch slice is too short reports `DimensionMismatch`.
//! Exceeding a configured memory or capacity allowance is `ResourceLimit`;
//! each workspace API specifies how it reports insufficient reserved storage.
//! Failed growth is `AllocationFailure`, and size arithmetic overflow is
//! `SizeOverflow`. Reservations may retain capacity gained before a failure.
//! Requirements account for retained capacity, including storage from earlier calls.
//!
//! # Execution contexts
//!
//! An [`ExecutionContext`] carries the resources a computation may use: an
//! [`Executor`] for parallel work, and optionally a clock for deadlines, a
//! [`Cancellation`] flag, an entropy source and the processor features that
//! kernels may use. SciOxide does not read any of them from global state. Work
//! started from inside a parallel callback runs on that callback's thread.
//! [`ExecutionContext::sequential`] runs on the calling thread with none of them.
//! A function without a context parameter runs with
//! [`ExecutionContext::sequential_native`], whose kernels may use the
//! processor's native features; [`ExecutionContext::native`] gives your
//! executor the same kernels.
//!
//! An [`ExecutionPolicy`] states how many workers an operation may use:
//! `Sequential`, `Parallel(n)`, or `Auto`, which goes parallel only when the
//! work is large enough. Work is split into contiguous partitions whose results
//! are combined in a fixed order, so for a given number of workers the result
//! does not depend on thread scheduling.
//! [`ThreadExecutor`] runs partitions on scoped standard-library threads; with
//! `Auto` it starts a worker only for at least 32,768 estimated work units.
//!
//! Cancellation and deadlines are cooperative. A computation checks them between
//! units of work and returns `ExecutionStopped`; it never abandons a running
//! callback.
//!
//! # Solve accounting
//!
//! Iterative solvers take a [`SolveContext`], obtained from
//! [`SolveState::begin_solve`] or through a [`SolveScope`]. It counts
//! iterations and callback evaluations in [`SolveCounts`] and enforces the
//! limits in [`SolveBudgets`]. A nested solve runs in a child scope: its
//! evaluations count toward the parent's evaluation limit, while its iterations
//! count only toward its own.
//!
//! A function whose name ends in `_scoped` takes the caller's scope or context
//! as its last argument and runs inside it. It observes the caller's
//! cancellation and deadline, and a nested solve counts toward the caller's
//! limits. The form without the suffix runs the same computation on its own.
//!
//! Each solver crate defines its own termination status. When a budget,
//! cancellation or deadline ends a solve that has a usable last iterate, the
//! solver returns `Ok` with an unsuccessful status. Otherwise it returns
//! `ConvergenceFailure` or `ExecutionStopped`.
//! Value-only one-shot forms cannot return an unsuccessful partial result and
//! translate it into an error. Progressive solvers preserve the last accepted
//! state and already emitted samples; their owned forms may return progress in a
//! crate-specific partial-failure type. Consult the entry point's return type.
//! Work-budget exhaustion is distinct from a storage `ResourceLimit`.
//!
//! # Allocation
//!
//! Functions that take a workspace or scratch buffer of sufficient size do not
//! allocate. Scalar functions such as [`checked_sum`], [`logsumexp`] and
//! [`sin_pi`] do not allocate either.
//! Convenience forms that return a `Vec`, such as [`softmax`], allocate their
//! result. An executor may allocate to dispatch work; that is separate from the
//! numerical kernels. Item docs mention allocation only when an item differs
//! from this rule.
//!
//! # Feature flags
//!
//! - `std` (default): enables [`ThreadExecutor`], [`StdClock`] and
//!   [`StdProcessor`], and runtime detection of CPU features.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. CPU
//! features are then taken from the compile-time target features, and
//! [`ExecutionContext::sequential_native`] uses [`PortableProcessor`].
//!
//! The [`backend`] module holds lower-level building blocks used by the other
//! SciOxide crates. Most users do not need it.
//!
//! # References
//!
//! - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//!   SIAM, 2002, chapter 4 (summation).
//! - T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product",
//!   *SIAM J. Sci. Comput.* 26(6), 2005.
//! - A. Neumaier, "Rundungsfehleranalyse einiger Verfahren zur Summation
//!   endlicher Summen", *ZAMM* 54, 1974.
//! - J. L. Blue, "A portable Fortran program to find the Euclidean norm of a
//!   vector", *ACM Trans. Math. Softw.* 4(1), 1978.
//! - M. H. Payne and R. N. Hanek, "Radian reduction for trigonometric
//!   functions", *ACM SIGNUM Newsletter* 18(1), 1983.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unreachable))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod accuracy;
mod affine;
mod arithmetic;
mod attempt;
pub mod backend;
mod bounds;
mod budget;
mod capacity;
mod combinatorics;
mod complex;
mod error;
mod error_identity;
mod execution;
mod identifier;
mod interval;
mod kernel;
mod layout;
mod logarithmic;
mod math;
mod nonlinear;
mod operator;
mod scalar;
mod scaled;
mod size;
mod solve;
mod stable;
mod stable_batch;
mod stable_complex;
mod structure;
mod tolerance;
mod value;

pub use accuracy::Accuracy;
pub use affine::{
    AffineValue, AffineValueMut, affine_combination_into, check_value_shape, validate_value,
    write_value,
};
pub use arithmetic::{
    CompensatedSum, HARDWARE_FMA, ProductSum, PromotingCompensatedSum, PromotingProduct,
    RangeAccumulator, ScaledNorm, checked_affine, checked_affine_ratio, checked_difference,
    checked_difference_of_products, checked_float_product, checked_float_product_factors,
    checked_float_quotient, checked_midpoint, checked_mul_add, checked_next_after,
    checked_product_quotient, checked_quadratic_roots, checked_scalbn, checked_scaled_norm,
    checked_sum, checked_sum_of_products, compensated_reduce, fast_mul_add,
};
pub use attempt::{AttemptKind, AttemptOutcome, AttemptRecord, PartialFailure};
pub use bounds::Bounds;
pub use budget::Budget;
pub use capacity::Capacity;
pub use combinatorics::{
    CombinationEnumerator, MultisetEnumerator, Permutation, PermutationEnumerator, PermutationView,
    checked_binomial, checked_double_factorial, checked_factorial, checked_multifactorial,
    checked_multiset_count, checked_permutation_count, checked_rising_factorial,
    checked_stirling_first_kind, checked_stirling_second_kind, combination_rank,
    combination_unrank_into, multiset_rank, multiset_unrank_into, permutation_rank,
    permutation_unrank_into, permutation_validation_scratch_bytes,
};
pub use complex::{Complex, Complex32, Complex64};

pub use error::{
    Capability, ConvergenceDiagnostics, ConvergenceDiagnosticsView, ConvergenceReason,
    DifferentiationFrame, DifferentiationLocation, DifferentiationSite, Dimensions, DomainReason,
    ExecutionFailureReason, ExecutionStopReason, InvalidInputReason, NestedLevel, NonFiniteOrigin,
    NumericalIssue, ScalarConversionReason, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
pub use error_identity::ConversionType;
pub use execution::{
    Cancellation, CancellationToken, CapabilityError, Clock, ClockBinding, Deadline, EntropySource,
    ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor, MonotonicInstant, PartitionFn,
    PartitionId, PlanLimits, PortableProcessor, ProcessorCapabilities, ProcessorFeature,
    SequentialExecutor, WorkPartition, WorkPartitions,
};
#[cfg(feature = "std")]
pub use execution::{StdClock, StdProcessor, ThreadExecutor};
pub use identifier::{AlgorithmComponent, Argument, IdentifierError, Quantity};
pub use interval::{ExtendedInterval, Interval};
pub use layout::{AxisLayout, BatchLayout, NdView, NdViewMut, Shape, StridedLayout};
pub use logarithmic::{logsumexp, logsumexp_value, weighted_logsumexp};
pub use nonlinear::{JacobianOperator, ResidualOperator};
pub use operator::{
    AdjointOperator, BlockOperator, BlockVectorView, BlockVectorViewMut, ClosureOperator,
    ClosurePreconditioner, DenseOperatorInput, DenseOperatorKernel, LinearOperator,
    OperatorContext, Preconditioner,
};
pub use scalar::{AlgebraicScalar, FloatDecomposition, NumericScalar, RealScalar};
pub use scaled::{Scaled, ScaledAccumulator, ScaledValue};
pub use solve::{
    AcceptedStateMetadata, AccountingIntegrity, Admission, AlgorithmId, BudgetKind, BudgetLimit,
    BudgetReport, EstimateReliability, EvaluationCounts, EvaluationRole, Invocation,
    PartialResultTrust, ProblemToken, ReliableEstimate, ResidualReport, ResidualScaling,
    ResidualScalingRule, SolveBudgets, SolveCheckpointMetadata, SolveContext, SolveCounts,
    SolveEpoch, SolveIdentity, SolveScope, SolveState,
};
pub use stable::{
    cos_degrees, cos_pi, cosm1, exprel, log_sigmoid, log1mexp, logaddexp, logit, powm1, sigmoid,
    sin_cos_degrees, sin_cos_pi, sin_degrees, sin_pi, sinc, sincpi, softplus, tan_degrees, tan_pi,
    xlog1py, xlogy,
};
pub use stable_batch::{
    log_softmax, log_softmax_axis, log_softmax_axis_into, log_softmax_into, logsumexp_axis,
    logsumexp_axis_into, softmax, softmax_axis, softmax_axis_into, softmax_into, transform_into,
    weighted_logsumexp_axis, weighted_logsumexp_axis_into,
};
pub use structure::{
    ColoringKind, ColoringSchedule, DiagonalKind, StructuralPattern, StructuralPatternView,
    Triangle,
};
pub use tolerance::{
    AbsoluteTolerance, ErrorTolerance, FeasibilityTolerance, GradientTolerance,
    IntegralityTolerance, NormSelection, PreparedErrorNorm, RankThreshold, RelativeTolerance,
    ResidualTolerance, StepTolerance, Tolerance,
};
pub use value::{LogValue, ProjectiveValue, ValueAndDerivative};

pub(crate) use identifier::ValidatedIdentifier;
