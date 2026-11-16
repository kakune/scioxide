//! Indexed zeros of Bessel, Airy, Kelvin, error and Fresnel functions.
//!
//! Starting values come from short tables of the first zeros and from
//! asymptotic expansions: McMahon and uniform (Olver) expansions for Bessel
//! zeros, the Airy expansions of DLMF 9.9, McMahon-type expansions for Kelvin
//! zeros (DLMF 10.70), and the expansions of Fettis, Caslin and Cramer for erf
//! and of DLMF 7.13 for Fresnel zeros. Real zeros are refined by safeguarded
//! Newton iteration in a sign bracket, complex zeros by Newton iteration in
//! the cell of their start. Each zero is then checked on the original
//! equation at its rounded output coordinate. Where no start applies, Bessel
//! zeros are found by bounded enumeration of sign changes.
use crate::RootTermination;
mod evaluation;
mod seeds;
mod solve;
mod starts;
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    Budget, Complex, EvaluationCounts, ExecutionPolicy, Interval, RealScalar, ResidualReport,
    SciOxideError, SolveContext, StepTolerance,
};
/// Selects the Kelvin function of order zero, or its derivative, for
/// [`SpecialZeroFamily::Kelvin`].
///
/// The Kelvin functions are `ber x + i bei x = J₀(x e^(3πi/4))` and
/// `ker x + i kei x = K₀(x e^(πi/4))` for real `x > 0` (DLMF 10.61.1–2).
///
/// ```
/// use scioxide_rootfind::{KelvinZeroFunction, SpecialZeroFamily, special_zeros};
///
/// // First positive zero of ber x, from a 45-digit reference value.
/// let zeros = special_zeros(SpecialZeroFamily::<f64>::Kelvin(KelvinZeroFunction::Ber), 1)?;
/// assert!((zeros[0] - 2.848917820795140).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KelvinZeroFunction {
    /// Zeros of `ber x`.
    Ber,
    /// Zeros of `bei x`.
    Bei,
    /// Zeros of `ker x`.
    Ker,
    /// Zeros of `kei x`.
    Kei,
    /// Zeros of `ber′ x`.
    BerPrime,
    /// Zeros of `bei′ x`.
    BeiPrime,
    /// Zeros of `ker′ x`.
    KerPrime,
    /// Zeros of `kei′ x`.
    KeiPrime,
}

/// Selects the function whose zeros [`special_zeros()`],
/// [`special_zeros_with_options`] and [`special_zeros_into`] compute.
///
/// Zeros are numbered from one in order of increasing magnitude; index zero
/// denotes a zero at the origin (see [`SpecialZeroBranch::Origin`]). Which
/// zeros are available depends on the [`SpecialZeroBranch`]:
///
/// | Family | `Real` branch | `Complex` branch | `Origin` |
/// |---|---|---|---|
/// | `BesselJ(ν)`, `ν > −1/2` | positive zeros `j_ν,k` | — | — |
/// | `BesselJPrime(ν)`, `ν ≥ 0` | positive zeros `j′_ν,k` | — | `ν = 0` |
/// | `BesselY(ν)`, `BesselYPrime(ν)`, `ν ≥ 0` | positive zeros | `ν ∈ {0, 1}`, conjugation | — |
/// | `AiryAi`, `AiryAiPrime` | negative zeros `aₖ`, `a′ₖ` | — | — |
/// | `AiryBi`, `AiryBiPrime` | negative zeros `bₖ`, `b′ₖ` | conjugation | — |
/// | `Kelvin(f)` | positive zeros | — | `bei`, `ber′`, `bei′` |
/// | `Erf` | none ([`RootTermination::NoZeros`]) | conjugation, negation | yes |
/// | `FresnelC`, `FresnelS` | none ([`RootTermination::NoZeros`]) | all symmetries | yes |
/// | `FresnelCPrime`, `FresnelSPrime` | positive zeros | all symmetries | `FresnelSPrime` |
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialZeroFamily<S: RealScalar> {
    /// Zeros of the Bessel function `J_ν(x)`, for finite `ν > −1/2`.
    BesselJ(S),
    /// Zeros of `J′_ν(x)`, for finite `ν ≥ 0`. The indexed sequence excludes
    /// `x = 0`.
    BesselJPrime(S),
    /// Zeros of the Bessel function `Y_ν(z)`, for finite `ν ≥ 0`. Complex zeros
    /// are available for `ν = 0` and `ν = 1`.
    BesselY(S),
    /// Zeros of `Y′_ν(z)`, for finite `ν ≥ 0`. Complex zeros are available for
    /// `ν = 0` and `ν = 1`.
    BesselYPrime(S),
    /// Zeros of the Airy function `Ai(x)`, all negative real.
    AiryAi,
    /// Zeros of `Ai′(x)`, all negative real.
    AiryAiPrime,
    /// Zeros of the Airy function `Bi(z)`: negative real zeros and complex
    /// zeros near the rays `arg z = ±π/3`.
    AiryBi,
    /// Zeros of `Bi′(z)`: negative real zeros and complex zeros near the rays
    /// `arg z = ±π/3`.
    AiryBiPrime,
    /// Positive real zeros of a Kelvin function of order zero.
    Kelvin(KelvinZeroFunction),
    /// Zeros of the error function `erf z`: the origin and complex zeros
    /// near the diagonals `arg z = ±π/4, ±3π/4`.
    Erf,
    /// Zeros of `C(z) = ∫₀ᶻ cos(πt²/2) dt`: the origin and complex zeros.
    FresnelC,
    /// Zeros of `C′(z) = cos(πz²/2)`, at `z = ±√(2k − 1)` and
    /// `z = ±i√(2k − 1)`.
    FresnelCPrime,
    /// Zeros of `S(z) = ∫₀ᶻ sin(πt²/2) dt`: the origin and complex zeros.
    FresnelS,
    /// Zeros of `S′(z) = sin(πz²/2)`, at the origin, `z = ±√(2k)` and
    /// `z = ±i√(2k)`.
    FresnelSPrime,
}
/// Selects which sequence of zeros [`special_zeros_with_options`] and
/// [`special_zeros_into`] compute; [`special_zeros()`] uses [`Self::Real`].
///
/// Zeros related by symmetry form separate sequences, so each index denotes
/// one zero. The supported combinations are listed on [`SpecialZeroFamily`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialZeroBranch {
    /// Real zeros: positive for Bessel, Kelvin and Fresnel derivative
    /// families, negative for Airy families.
    Real,
    /// Complex zeros. The base sequence lies in the upper half-plane (for
    /// Fresnel derivative families, on the positive real axis); each zero is
    /// then conjugated, negated and multiplied by `i`, in that order, as
    /// selected. `Y` and `Bi` families allow only conjugation, `Erf` allows
    /// conjugation and negation, and Fresnel families allow all three.
    Complex {
        /// Replaces `z` by its conjugate.
        conjugate: bool,
        /// Replaces `z` by `−z`, after conjugation.
        negate: bool,
        /// Replaces `z` by `iz`, after the other operations (Fresnel families
        /// only).
        rotate: bool,
    },
    /// The zero at the origin, as index zero. Requires `start = 0` and
    /// `count ≤ 1`.
    Origin,
}
impl SpecialZeroBranch {
    /// Selects the complex zeros in the upper half-plane, without
    /// transformation.
    pub const UPPER: Self = Self::Complex {
        conjugate: false,
        negate: false,
        rotate: false,
    };
}
/// Holds the tolerance and work limits of [`special_zeros_with_options`] and
/// [`special_zeros_into`]; [`special_zeros()`] uses the defaults.
///
/// | Option | Default |
/// |---|---|
/// | [`Self::with_tolerance`] | `atol = rtol = √ε` |
/// | [`Self::with_evaluation_budget`] | 2²⁶ function evaluations per solve |
/// | [`Self::with_work_per_zero`] | 2²⁰ iterations per zero |
/// | [`Self::with_refinement_steps`] | 100 Newton steps per zero |
/// | [`Self::with_budgets`] | unbounded |
/// | [`Self::with_execution_policy`] | sequential |
/// | [`Self::with_workspace_byte_limit`] | `usize::MAX` |
#[derive(Debug, Clone)]
pub struct SpecialZeroOptions<S: RealScalar> {
    tolerance: StepTolerance<S>,
    evaluation_budget: Budget,
    work_per_zero: Budget,
    refinement_steps: Budget,
    policy: ExecutionPolicy,
    budgets: scioxide_core::SolveBudgets,
    byte_limit: usize,
}
impl<S: RealScalar> Default for SpecialZeroOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: StepTolerance::default(),
            evaluation_budget: Budget::ONE.saturating_add((1 << 26) - 1),
            work_per_zero: Budget::ONE.saturating_add((1 << 20) - 1),
            refinement_steps: Budget::ONE.saturating_add(99),
            policy: ExecutionPolicy::Sequential,
            budgets: scioxide_core::SolveBudgets::unbounded(),
            byte_limit: usize::MAX,
        }
    }
}
impl<S: RealScalar> SpecialZeroOptions<S> {
    /// Sets the iteration and evaluation limits of the whole solve.
    ///
    /// The limits of a parent [`scioxide_core::SolveContext`] also apply (see
    /// [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn with_budgets(mut self, budgets: scioxide_core::SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the iteration and evaluation limits of the whole solve.
    pub const fn budgets(&self) -> scioxide_core::SolveBudgets {
        self.budgets
    }
    /// Sets the largest workspace, in bytes, that
    /// [`SpecialZeroWorkspace::workspace_requirements`] may report or a
    /// workspace may hold.
    pub const fn with_workspace_byte_limit(mut self, limit: usize) -> Self {
        self.byte_limit = limit;
        self
    }
    /// Returns the workspace byte limit.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }

    /// Sets the tolerance on the error of each zero.
    ///
    /// A zero `z` is accepted when its [`SpecialZero::estimated_error`], which
    /// includes the error of the function evaluation, is at most
    /// `atol + rtol |z|`.
    pub const fn with_tolerance(mut self, tolerance: StepTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Sets the limit on function evaluations of the whole solve, counting
    /// bracketing, refinement and the final check.
    pub const fn with_evaluation_budget(mut self, budget: Budget) -> Self {
        self.evaluation_budget = budget;
        self
    }
    /// Sets the iteration limit of the search for each zero, which also
    /// bounds the work of each special-function evaluation. Newton refinement
    /// has its own limit, [`Self::with_refinement_steps`].
    pub const fn with_work_per_zero(mut self, budget: Budget) -> Self {
        self.work_per_zero = budget;
        self
    }
    /// Sets the limit on Newton steps when refining each zero.
    pub const fn with_refinement_steps(mut self, budget: Budget) -> Self {
        self.refinement_steps = budget;
        self
    }
    /// Sets the execution policy. Zeros are computed independently, so a
    /// parallel policy distributes them over the executor of the execution
    /// context (see [Execution contexts](scioxide_core#execution-contexts)).
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the tolerance on the error of each zero.
    pub const fn tolerance(&self) -> StepTolerance<S> {
        self.tolerance
    }
}
/// Describes one computed zero and its error estimate.
///
/// Every returned zero has passed a check on the original equation `f(z) = 0`
/// at the returned coordinate. The error estimate is a numerical estimate, not
/// a rigorous bound; for guaranteed enclosures use `scioxide-validated`.
#[derive(Debug, Clone, Copy)]
pub struct SpecialZero<S: RealScalar> {
    index: usize,
    root: Complex<S>,
    residual: ResidualReport<S>,
    correction: S,
    estimated_error: S,
    inverse_slope: Option<S>,
    bracket: Option<Interval<S>>,
    multiplicity: usize,
    scaled: bool,
}
impl<S: RealScalar> SpecialZero<S> {
    /// Returns `ResidualChecked`: the zero passed a residual check, which is
    /// not a rigorous enclosure.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        scioxide_core::PartialResultTrust::ResidualChecked
    }

    /// Returns the index `k` of the zero, or zero for the origin.
    pub const fn index(&self) -> usize {
        self.index
    }
    /// Returns the zero, after the branch transformation and rounding to `S`.
    pub const fn point(&self) -> Complex<S> {
        self.root
    }
    /// Returns the residual `|f(z)|` at the returned zero.
    ///
    /// For Kelvin families `f` is exponentially scaled; see
    /// [`Self::residual_is_scaled`].
    pub const fn residual_report(&self) -> ResidualReport<S> {
        self.residual
    }
    /// Returns `|f(z)|` at the returned zero.
    pub const fn residual_norm(&self) -> S {
        self.residual.raw_norm()
    }
    /// Returns the size of the Newton correction `|f(z)/f′(z)|` at the
    /// returned zero, or zero for a zero at the origin.
    pub const fn correction(&self) -> S {
        self.correction
    }
    /// Returns the estimated error of the zero:
    ///
    /// ```text
    /// (|f(z)| + δf) / |f′(z)| + |z − z₆₄|,
    /// ```
    ///
    /// where `δf` bounds the error of evaluating `f(z)` and `z₆₄` is the
    /// zero computed in `f64` before rounding to `S`.
    pub const fn estimated_error(&self) -> S {
        self.estimated_error
    }
    /// Returns `1/|f′(z)|`, the absolute condition number of the zero, or
    /// `None` for a multiple zero at the origin or when the value overflows.
    pub const fn inverse_slope(&self) -> Option<S> {
        self.inverse_slope
    }
    /// Returns the interval in which `f` changes sign around a real zero, if
    /// the zero was refined in a bracket.
    ///
    /// The endpoints are rounded outward to `S`. The signs are computed in
    /// floating point, so the bracket is not a rigorous enclosure. For Airy
    /// families an endpoint sign may come from a Taylor polynomial of the Airy
    /// equation with a bounded remainder.
    pub const fn bracket(&self) -> Option<Interval<S>> {
        self.bracket
    }
    /// Returns the multiplicity: one for every nonzero zero, and the known
    /// multiplicity for a zero at the origin (for example three for
    /// `FresnelS`).
    pub const fn multiplicity(&self) -> usize {
        self.multiplicity
    }
    /// Returns whether the residual is that of an exponentially scaled
    /// function. This is `true` for Kelvin families, whose zeros are computed
    /// from the scaled functions of `scioxide-special`, such as
    /// `ber x · exp(−x/√2)` and `ker x · exp(x/√2)`.
    pub const fn residual_is_scaled(&self) -> bool {
        self.scaled
    }
}
/// Reports the outcome and work of [`special_zeros_with_options`] or
/// [`special_zeros_into`].
///
/// [`Self::termination`] is the outcome:
///
/// - [`RootTermination::Complete`]: every requested zero was found and
///   checked.
/// - [`RootTermination::NoZeros`]: the requested sequence is empty, for
///   example the real zeros of `erf`.
/// - [`RootTermination::Unverified`]: a zero failed the final check, or its
///   magnitude did not exceed that of the previous zero.
/// - [`RootTermination::IterationLimit`], [`RootTermination::WorkLimit`],
///   [`RootTermination::Cancelled`], [`RootTermination::TimeLimit`]: a limit,
///   cancellation or deadline stopped the search (see
///   [Termination](crate#termination)).
///
/// Zeros before the first failure are returned; [`Self::completed`] counts
/// them, and [`Self::candidate`] gives the last estimate of the zero that
/// failed.
#[derive(Debug, Clone)]
pub struct SpecialZeroReport<S: RealScalar> {
    algorithm: scioxide_core::AlgorithmId,
    termination: RootTermination,
    completed: usize,
    requested: usize,
    counts: scioxide_core::SolveCounts,
    candidate: Option<Complex<S>>,
}
impl<S: RealScalar> SpecialZeroReport<S> {
    /// Returns the algorithm identifier.
    pub const fn algorithm(&self) -> scioxide_core::AlgorithmId {
        self.algorithm
    }

    /// Returns the outcome of the solve.
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }
    /// Returns whether every requested zero was found, or the requested
    /// sequence is empty.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the iterations of the solve itself. The search for each zero
    /// runs as a nested solve, counted in [`Self::accounting`].
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns the number of zeros found, counted from the first requested
    /// index.
    pub const fn completed(&self) -> usize {
        self.completed
    }
    /// Returns the number of requested zeros.
    pub const fn requested(&self) -> usize {
        self.requested
    }
    /// Returns the iterations and evaluations of the solve itself. The search
    /// for each zero runs as a nested solve, so this is usually empty; see
    /// [`Self::accounting`].
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of the solve and of its nested per-zero solves.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the last estimate of the first zero that was not found, if
    /// there is one. It has not passed the final check.
    pub const fn candidate(&self) -> Option<Complex<S>> {
        self.candidate
    }
}
/// Holds the zeros and report returned by [`special_zeros_with_options`].
#[derive(Debug, Clone)]
pub struct SpecialZeros<S: RealScalar> {
    roots: Vec<SpecialZero<S>>,
    report: SpecialZeroReport<S>,
}
impl<S: RealScalar> SpecialZeros<S> {
    /// Returns the zeros found, in index order.
    pub fn roots(&self) -> &[SpecialZero<S>] {
        &self.roots
    }
    /// Returns the solve report.
    pub const fn report(&self) -> &SpecialZeroReport<S> {
        &self.report
    }
}
/// The zeros with indices `start, …, start + count − 1` of `family` on
/// `branch`, computed under `options`.
#[derive(Debug)]
struct Request<'a, S: RealScalar> {
    family: SpecialZeroFamily<S>,
    branch: SpecialZeroBranch,
    start: usize,
    count: usize,
    options: &'a SpecialZeroOptions<S>,
}
/// Computes the first `count` real zeros of a Bessel, Airy or Kelvin
/// function, or of the derivative of a Fresnel integral, in order of
/// increasing magnitude.
///
/// The result holds the real parts of the zeros with indices `1, …, count` on
/// the [`SpecialZeroBranch::Real`] branch of `family`: the positive zeros
/// `j_ν,1 < j_ν,2 < …` of `J_ν` and the positive zeros of the other Bessel,
/// Kelvin and Fresnel-derivative families, and the negative zeros
/// `a₁ > a₂ > …` of the Airy families (see [`SpecialZeroFamily`]). `erf`, `C`
/// and `S` have no real zeros other than the origin, so their result is empty,
/// as is the result for `count = 0`.
///
/// Each zero starts from a tabulated value or an asymptotic expansion in its
/// index and is refined in `f64` by Newton iteration safeguarded by bisection
/// in a sign bracket, then rounded to `S`. A zero is accepted when its
/// [`SpecialZero::estimated_error`] is at most `√ε·(1 + |z|)`, where `ε` is the
/// machine epsilon of `S` and `z` the zero, within 100 Newton steps and 2²⁰
/// iterations per zero and 2²⁶ function evaluations in all. Each zero costs a
/// few evaluations of the function and its derivative; the call allocates the
/// result and a workspace of `O(count)` storage and runs sequentially. The
/// values equal the real parts of the zeros of [`special_zeros_with_options`]
/// on the `Real` branch from index 1 with the default [`SpecialZeroOptions`].
///
/// Use [`special_zeros_with_options`] for complex zeros, the zero at the
/// origin, a later first index, other tolerances or limits, or the error
/// estimates of each zero, and [`special_zeros_into`] to reuse storage or run
/// in an execution context.
///
/// # Errors
///
/// - `InvalidInput` (`family`, `OutOfRange`) if the order `ν` is out of range
///   (`ν ≤ −1/2` for `BesselJ`, `ν < 0` for the other Bessel families), and
///   `ScalarConversion` if it is NaN or infinite.
/// - `InvalidInput` (`count`, `OutOfRange`) if `1 + count > 2⁵²`, and
///   `SizeOverflow` if `1 + count` overflows `usize`.
/// - `ResourceLimit`, `SizeOverflow` or `AllocationFailure` if the workspace or
///   result cannot be created.
/// - `ConvergenceFailure` if a limit stops the search (`IterationLimit`,
///   `EvaluationLimit`) or a zero fails its final check (`Stagnation`); see the
///   [crate errors](crate#errors).
/// - An error from a special-function evaluation, such as
///   `NumericalInstability`.
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{SpecialZeroFamily, special_zeros};
///
/// // The first three positive zeros j₀,₁ < j₀,₂ < j₀,₃ of the Bessel function J₀.
/// let zeros = special_zeros(SpecialZeroFamily::BesselJ(0.0_f64), 3)?;
/// assert_eq!(zeros.len(), 3);
/// assert!((zeros[0] - 2.404_825_557_695_773).abs() < 1e-12);
/// assert!((zeros[1] - 5.520_078_110_286_311).abs() < 1e-12);
/// assert!((zeros[2] - 8.653_727_912_911_013).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §7.13 (zeros of the Fresnel integrals), §9.9 (Airy zeros),
/// §10.21 (Bessel zeros), §10.68 and §10.70 (Kelvin functions).
/// <https://dlmf.nist.gov/>
pub fn special_zeros<S: RealScalar>(
    family: SpecialZeroFamily<S>,
    count: usize,
) -> Result<Vec<S>, SciOxideError> {
    let zeros = special_zeros_with_options(
        family,
        SpecialZeroBranch::Real,
        1,
        count,
        &SpecialZeroOptions::default(),
    )?;
    let report = zeros.report();
    crate::termination::require_success(report.termination(), report.accounting())?;
    let mut values = try_with_capacity(zeros.roots().len(), scioxide_core::quantity!("zeros"))?;
    values.extend(zeros.roots().iter().map(|zero| zero.point().re));
    Ok(values)
}
/// Computes consecutive indexed zeros of a Bessel, Airy, Kelvin, error or
/// Fresnel function and returns them with a report.
///
/// The result holds the zeros with indices `start, start + 1, …,
/// start + count − 1` of `family` on `branch`. Zeros are numbered from one in
/// order of increasing magnitude, as in DLMF; for example `j_ν,1` is the
/// smallest positive zero of `J_ν`. Index zero is the zero at the origin
/// ([`SpecialZeroBranch::Origin`]).
///
/// Each zero starts from a tabulated value or an asymptotic expansion in its
/// index. Real zeros are refined by Newton iteration safeguarded by bisection
/// in a sign bracket; complex zeros by Newton iteration that must stay in the
/// cell of their starting value. Zeros are computed in `f64` and rounded to
/// `S`. Each zero is then checked on the original equation at its returned
/// coordinate: rounding must move it by less than 0.45 times the distance
/// between consecutive starting values, so that it still identifies the
/// indexed zero, and its [`SpecialZero::estimated_error`] must be at most
/// `atol + rtol·|z|` ([`SpecialZeroOptions::with_tolerance`]; by default
/// `atol = rtol = √ε`, with `ε` the machine epsilon of `S`). The magnitudes of
/// consecutive zeros must increase.
///
/// The call allocates the output and a workspace of `O(count)` storage and
/// runs sequentially. When a limit stops the search or a zero fails its check,
/// the zeros found before it are returned and the report has an unsuccessful
/// termination (see [`SpecialZeroReport`]); `count = 0` gives an empty result.
/// [`special_zeros()`] returns the real zeros from index 1 with the default
/// options, and [`special_zeros_into`] writes the zeros into caller storage
/// with a prepared workspace.
///
/// # Errors
///
/// - `InvalidInput` (`family`, `OutOfRange`) if the order `ν` is out of range
///   (`ν ≤ −1/2` for `BesselJ`, `ν < 0` for the other Bessel families), and
///   `ScalarConversion` if it is NaN or infinite.
/// - `InvalidInput` (`start`, `OutOfRange`) if `start` is zero on the `Real` or
///   `Complex` branch.
/// - `InvalidInput` (`branch`, `OutOfRange`) if the `Origin` branch is used
///   with `start ≠ 0`, `count > 1` or a family without a zero at the origin,
///   or if the family does not support the requested complex zeros or
///   symmetry (see [`SpecialZeroFamily`]).
/// - `InvalidInput` (`count`, `OutOfRange`) if `start + count` exceeds 2⁵²,
///   and `SizeOverflow` if it overflows `usize`.
/// - `ResourceLimit`, `SizeOverflow` or `AllocationFailure` if the workspace or
///   output cannot be created.
/// - An error from a special-function evaluation that is not a stop at a
///   limit, such as `NumericalInstability`.
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{
///     SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions, special_zeros_with_options,
/// };
///
/// // J_{1/2}(x) = √(2/(πx)) sin x, so its positive zeros are kπ; request k = 3 and 4.
/// let family = SpecialZeroFamily::BesselJ(0.5);
/// let options = SpecialZeroOptions::default();
/// let zeros = special_zeros_with_options(family, SpecialZeroBranch::Real, 3, 2, &options)?;
/// assert!(zeros.report().is_success());
/// for (zero, k) in zeros.roots().iter().zip([3.0, 4.0]) {
///     assert!((zero.point().re - k * core::f64::consts::PI).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §7.13 (zeros of the error and Fresnel functions), §9.9 (Airy
///   zeros), §10.21 (Bessel zeros), §10.68 and §10.70 (Kelvin functions).
///   <https://dlmf.nist.gov/>
/// - H. E. Fettis, J. C. Caslin and K. R. Cramer, "Complex zeros of the error
///   function and of the complementary error function", *Math. Comp.* 27
///   (1973), 401–407.
pub fn special_zeros_with_options<S: RealScalar>(
    family: SpecialZeroFamily<S>,
    branch: SpecialZeroBranch,
    start: usize,
    count: usize,
    options: &SpecialZeroOptions<S>,
) -> Result<SpecialZeros<S>, SciOxideError> {
    let request = Request {
        family,
        branch,
        start,
        count,
        options,
    };
    solve::owned(
        &request,
        [
            scioxide_core::argument!("family"),
            scioxide_core::argument!("branch"),
            scioxide_core::argument!("start"),
            scioxide_core::argument!("count"),
        ],
    )
}
/// Computes the zeros with indices `start, …, start + output.len() − 1` into
/// `output` with a prepared workspace.
///
/// The zeros, their checks and the report are those of
/// [`special_zeros_with_options`] with `count = output.len()`: `output[i]`
/// receives the zero with index `start + i`. If the solve stops early, the
/// first [`SpecialZeroReport::completed`] slots receive zeros and the remaining
/// slots are not written. Work counts toward the budgets of `scope` (see
/// [Solve accounting](scioxide_core#solve-accounting)), and a parallel
/// [`SpecialZeroOptions::with_execution_policy`] distributes the zeros over
/// the executor of `scope`.
///
/// Create the workspace with [`SpecialZeroWorkspace::workspace`], or size it
/// with [`SpecialZeroWorkspace::workspace_requirements`]; its size depends
/// only on the number of zeros. With a prepared workspace the call does not
/// allocate.
///
/// # Errors
///
/// - The `InvalidInput`, `ScalarConversion` and `SizeOverflow` errors of
///   [`special_zeros_with_options`] for `family`, `branch` and `start`, naming
///   `output` where that function names `count`.
/// - `InvalidInput` (`workspace`, `OutOfRange`) if `workspace` cannot hold
///   `output.len()` zeros.
/// - `ResourceLimit` or `SizeOverflow` if the workspace size for
///   `output.len()` zeros exceeds [`SpecialZeroOptions::workspace_byte_limit`]
///   or overflows `usize`.
/// - The evaluation errors of [`special_zeros_with_options`], and errors of the
///   execution context of `scope`.
///
/// On error, `output` is unchanged (see
/// [Output on error](scioxide_core#output-on-error)).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_rootfind::{SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions};
/// use scioxide_rootfind::{SpecialZeroWorkspace, special_zeros_into};
///
/// // The zeros of J_{1/2}(x) = √(2/(πx)) sin x are kπ; compute k = 5, …, 8.
/// let (family, branch) = (SpecialZeroFamily::BesselJ(0.5), SpecialZeroBranch::Real);
/// let options = SpecialZeroOptions::default();
/// let mut ws = SpecialZeroWorkspace::workspace(4, &options)?;
/// let mut zeros = [None; 4];
/// let cx = ExecutionContext::sequential_native();
/// assert!(special_zeros_into(family, branch, 5, &options, &mut zeros, &mut ws, cx)?.is_success());
/// for (zero, k) in zeros.iter().flatten().zip([5.0, 6.0, 7.0, 8.0]) {
///     assert!((zero.point().re - k * core::f64::consts::PI).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn special_zeros_into<'scope, 'capability, S: RealScalar>(
    family: SpecialZeroFamily<S>,
    branch: SpecialZeroBranch,
    start: usize,
    options: &SpecialZeroOptions<S>,
    output: &mut [Option<SpecialZero<S>>],
    workspace: &mut SpecialZeroWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<SpecialZeroReport<S>, SciOxideError> {
    let request = Request {
        family,
        branch,
        start,
        count: output.len(),
        options,
    };
    solve::run(
        &request,
        [
            scioxide_core::argument!("family"),
            scioxide_core::argument!("branch"),
            scioxide_core::argument!("start"),
            scioxide_core::argument!("output"),
        ],
        output,
        workspace,
        scope.into(),
    )
}

/// Describes the workspace that [`special_zeros_into`] needs for a number of
/// zeros; see [`SpecialZeroWorkspace::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct SpecialZeroWorkspaceRequirements<S: RealScalar> {
    scalar: core::marker::PhantomData<fn() -> S>,
    capacity: usize,
    bytes: usize,
    limit: usize,
}
impl<S: RealScalar> SpecialZeroWorkspaceRequirements<S> {
    /// Returns the workspace size in bytes.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the number of zeros the workspace must hold.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}
fn special_bytes<S: RealScalar>(capacity: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(
        capacity,
        core::mem::size_of::<scioxide_core::backend::PartitionedSolve>()
            + core::mem::size_of::<Option<Result<solve::Attempt<S>, SciOxideError>>>(),
        scioxide_core::quantity!("workspace_bytes"),
    )
}
/// Holds the scratch storage of [`special_zeros_into`] and the record of the
/// last solve.
///
/// [`Self::workspace`] creates a workspace for a number of zeros, and
/// [`Self::workspace_requirements`] gives its size. The general rules are in
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct SpecialZeroWorkspace<S: RealScalar> {
    children: Vec<scioxide_core::backend::PartitionedSolve>,
    outcomes: Vec<Option<Result<solve::Attempt<S>, SciOxideError>>>,
    shell: crate::attempt::Shell<SpecialZeroReport<S>>,
}
impl<S: RealScalar> Default for SpecialZeroWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> SpecialZeroWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            children: Vec::new(),
            outcomes: Vec::new(),
            shell: crate::attempt::Shell::new(),
        }
    }
    /// Returns the size of a workspace for `count` zeros under `options`.
    ///
    /// The workspace holds one slot per zero, so its size grows linearly with
    /// `count`; the family, branch and first index do not enter it. `count = 0`
    /// gives an empty workspace.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds
    ///   [`SpecialZeroOptions::workspace_byte_limit`].
    pub fn workspace_requirements(
        count: usize,
        options: &SpecialZeroOptions<S>,
    ) -> Result<SpecialZeroWorkspaceRequirements<S>, SciOxideError> {
        let bytes = special_bytes::<S>(count)?;
        if bytes > options.byte_limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                options.byte_limit,
            ));
        }
        Ok(SpecialZeroWorkspaceRequirements {
            scalar: core::marker::PhantomData,
            capacity: count,
            bytes,
            limit: options.byte_limit,
        })
    }
    /// Allocates a workspace with which [`special_zeros_into`] computes up to
    /// `count` zeros under `options`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`
    /// if the storage cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_rootfind::{SpecialZeroOptions, SpecialZeroWorkspace};
    ///
    /// let options = SpecialZeroOptions::<f64>::default();
    /// let workspace = SpecialZeroWorkspace::workspace(16, &options)?;
    /// let requirements = SpecialZeroWorkspace::workspace_requirements(8, &options)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// assert_eq!(workspace.capacity(), 16);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(count: usize, options: &SpecialZeroOptions<S>) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(count, options)?)?;
        Ok(workspace)
    }
    /// Returns whether the workspace can hold the zeros of `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the storage already held, grown to the request,
    ///   would exceed the byte limit.
    pub fn check(
        &self,
        requirements: &SpecialZeroWorkspaceRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let q = scioxide_core::quantity!("workspace_bytes");
        let bytes = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_mul(
                self.children.capacity().max(requirements.capacity),
                core::mem::size_of::<scioxide_core::backend::PartitionedSolve>(),
                q,
            )?,
            scioxide_core::backend::checked_mul(
                self.outcomes.capacity().max(requirements.capacity),
                core::mem::size_of::<Option<Result<solve::Attempt<S>, SciOxideError>>>(),
                q,
            )?,
            q,
        )?;
        if bytes > requirements.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                requirements.limit,
            ));
        }
        Ok(
            if self.children.len() >= requirements.capacity
                && self.outcomes.len() >= requirements.capacity
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to hold the zeros of `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit`, as for [`Self::check`].
    /// - `AllocationFailure` if the storage cannot be allocated. Storage
    ///   allocated before the failure is kept.
    pub fn try_reserve(
        &mut self,
        requirements: &SpecialZeroWorkspaceRequirements<S>,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let q = scioxide_core::quantity!("special_zeros_workspace");
        let capacity = requirements.capacity;
        let additional = capacity.saturating_sub(self.children.len());
        scioxide_core::backend::try_reserve(&mut self.children, additional, q)?;
        self.children.resize_with(capacity, Default::default);
        let additional = capacity.saturating_sub(self.outcomes.len());
        scioxide_core::backend::try_reserve(&mut self.outcomes, additional, q)?;
        self.outcomes.resize_with(capacity, || None);
        let _ = self.check(requirements)?;
        Ok(())
    }
    /// Returns the number of zeros the workspace can hold without growing.
    pub fn capacity(&self) -> usize {
        self.children.len().min(self.outcomes.len())
    }
    /// Returns the record of the most recent call, including a call rejected
    /// before any numerical work.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<SpecialZeroReport<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the report of the most recent call if it found at least one
    /// zero, or `None` otherwise. A report from an earlier call is never
    /// returned.
    pub fn last_accepted(&self) -> Option<&SpecialZeroReport<S>> {
        self.shell.progress()
    }
}
crate::attempt::attempted!(SpecialZeroWorkspace<S>, [S: RealScalar], SpecialZeroReport<S>);
