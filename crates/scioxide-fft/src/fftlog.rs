//! Hamilton FFTLog with explicit logarithmic grids and complex bias.
use crate::axis::AxisPlan;
use crate::complex::ComplexCore;
use crate::memory::admit_bytes;
use crate::resource::Work;
use crate::scaling::{BinaryScale, complex_max, range_exponent};
use crate::validation::numerical;
use crate::validation::{invalid, length};
use crate::{FftBackend, FftPlanningContext};
use alloc::vec::Vec;
use core::f64::consts::LN_2;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Budget, Complex, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError, SolveContext,
};
type C = Complex<f64>;

/// Validated `f64` grid and accepted outer storage, before child preparation.
struct PreparationRequest {
    n: usize,
    count: f64,
    bias: C,
    spacing: f64,
    offset: f64,
    order: f64,
    block: usize,
    outer_bytes: usize,
    staging_bytes: usize,
}

struct PreparedFft<S: RealScalar> {
    fft: ComplexCore<S>,
    scratch: usize,
}

/// Log-domain Mellin coefficients, with direction-specific singularity and range.
struct MellinSpectrum {
    logs: Vec<MellinCoefficient>,
    maximum: f64,
    minimum: f64,
    forward_singular: bool,
    inverse_singular: bool,
}

struct PreparedCoefficients<S: RealScalar> {
    values: Vec<Complex<S>>,
    exponent: f64,
    condition: f64,
    singular: bool,
}

struct PreparedWeights<S: RealScalar> {
    radial: BiasWeights<S>,
    frequency: BiasWeights<S>,
}

/// Sets the storage and work limits of an [`FftLogPlan`].
///
/// Every limit defaults to unbounded. Exceeding a byte limit is a
/// `ResourceLimit` error, and exhausting `preparation_work` is a
/// `ConvergenceFailure`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{FftDirection, FftLogLimits, FftLogPlanSpec};
///
/// let limits = FftLogLimits {
///     workspace_bytes: 1 << 20,
///     ..FftLogLimits::default()
/// };
/// let bias = Complex::new(0.0_f64, 0.0);
/// let spec = FftLogPlanSpec::new(8, 0.1, 0.0, 0.0, bias, FftDirection::Forward);
/// assert_eq!(spec.with_limits(limits).limits().workspace_bytes, 1 << 20);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FftLogLimits {
    /// Maximum bytes live at once during preparation, excluding allocator metadata.
    pub preparation_bytes: usize,
    /// Maximum bytes of an execution workspace.
    pub workspace_bytes: usize,
    /// Maximum preparation work, counted in function calls and log-gamma iterations.
    pub preparation_work: Budget,
}
impl Default for FftLogLimits {
    fn default() -> Self {
        Self {
            preparation_bytes: usize::MAX,
            workspace_bytes: usize::MAX,
            preparation_work: Budget::ONE.saturating_add(usize::MAX),
        }
    }
}

/// Describes an FFTLog transform: grid, Bessel order, offset, bias and direction.
///
/// The parameters are defined on [`FftLogPlan`]. A spec is not validated until
/// a plan is prepared from it.
#[derive(Debug, Clone, Copy)]
pub struct FftLogPlanSpec<S: RealScalar> {
    direction: crate::FftDirection,
    /// Positive sample count `n`.
    pub(crate) length: usize,
    /// Real Bessel order `μ`.
    pub(crate) order: S,
    /// Positive logarithmic sample spacing `Δ`.
    pub(crate) log_spacing: S,
    /// Natural logarithm of the central input radius `r_c`.
    pub(crate) log_radius_center: S,
    /// Offset `L = ln(r_c k_c)`.
    pub(crate) offset: S,
    /// Complex bias exponent `q`.
    pub(crate) bias: Complex<S>,
    /// Execution policy of the FFT.
    pub(crate) execution_policy: ExecutionPolicy,
    /// Preparation and workspace limits.
    pub(crate) limits: FftLogLimits,
    iteration_budget: Option<scioxide_core::Budget>,
    backend_strategy: crate::FftBackendStrategy,
}
impl<S: RealScalar> FftLogPlanSpec<S> {
    /// Creates a spec for `length` samples with spacing `Δ = log_spacing`, Bessel
    /// order `μ = order`, offset `L = offset`, bias `q = bias` and the given
    /// direction.
    ///
    /// The central input radius is `r_c = 1` unless
    /// [`Self::with_log_radius_center`] moves it. The spec runs sequentially with
    /// automatic backend selection, no iteration budget and unbounded limits.
    ///
    /// # Examples
    ///
    /// A plan with real bias agrees with the one-call [`fht`](crate::fht).
    ///
    /// ```
    /// use scioxide_core::{Complex, ExecutionContext};
    /// use scioxide_fft::{FftDirection, FftLogPlan, FftLogPlanSpec, fht};
    ///
    /// let a = [1.0_f64, -0.5, 2.0, 0.25, 0.75];
    /// let expected = fht(&a, 0.3, 0.5, 0.1, 0.2)?;
    /// let bias = Complex::new(0.2, 0.0);
    /// let spec = FftLogPlanSpec::new(a.len(), 0.3, 0.5, 0.1, bias, FftDirection::Forward);
    /// let plan = FftLogPlan::new(spec)?;
    /// let input = a.map(|x| Complex::new(x, 0.0));
    /// let mut output = [Complex::new(0.0, 0.0); 5];
    /// let cx = ExecutionContext::sequential_native();
    /// plan.execute_into(&input, &mut output, &mut plan.workspace(())?, cx)?;
    /// for (z, x) in output.iter().zip(expected) {
    ///     assert!((z.re - x).abs() < 1e-12);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        length: usize,
        log_spacing: S,
        order: S,
        offset: S,
        bias: Complex<S>,
        direction: crate::FftDirection,
    ) -> Self {
        Self {
            length,
            log_spacing,
            order,
            log_radius_center: S::ZERO,
            offset,
            bias,
            direction,
            execution_policy: ExecutionPolicy::Sequential,
            backend_strategy: crate::FftBackendStrategy::Automatic,
            limits: FftLogLimits::default(),
            iteration_budget: None,
        }
    }
    /// Returns the sample count `n`.
    pub const fn length(&self) -> usize {
        self.length
    }
    /// Returns the logarithmic sample spacing `Δ`.
    pub const fn log_spacing(&self) -> S {
        self.log_spacing
    }
    /// Returns the Bessel order `μ`.
    pub const fn order(&self) -> S {
        self.order
    }
    /// Returns the offset `L`.
    pub const fn offset(&self) -> S {
        self.offset
    }
    /// Returns the bias exponent `q`.
    pub const fn bias(&self) -> Complex<S> {
        self.bias
    }
    /// Returns the transform direction.
    pub const fn direction(&self) -> crate::FftDirection {
        self.direction
    }
    /// Returns `ln r_c`, the natural logarithm of the central input radius.
    pub const fn log_radius_center(&self) -> S {
        self.log_radius_center
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the iteration budget of each execution, if any.
    pub const fn iteration_budget(&self) -> Option<scioxide_core::Budget> {
        self.iteration_budget
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> crate::FftBackendStrategy {
        self.backend_strategy
    }
    /// Returns the storage and work limits.
    pub const fn limits(&self) -> FftLogLimits {
        self.limits
    }

    /// Sets the central input radius to `r_c = exp(log_radius_center)`.
    ///
    /// The default is `r_c = 1`. The center moves the coordinates reported by
    /// [`FftLogPlan::log_radius`] and [`FftLogPlan::log_frequency`]. The
    /// transform values depend on the grid only through the offset `L`.
    pub const fn with_log_radius_center(mut self, log_radius_center: S) -> Self {
        self.log_radius_center = log_radius_center;
        self
    }
    /// Sets the CPU backend strategy of the FFT, checked when the plan is prepared.
    pub const fn with_backend_strategy(mut self, strategy: crate::FftBackendStrategy) -> Self {
        self.backend_strategy = strategy;
        self
    }
    /// Sets the execution policy of the FFT.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
    /// Limits the iterations of each execution; each transformed lane costs
    /// `length` iterations.
    pub const fn with_iteration_budget(mut self, budget: scioxide_core::Budget) -> Self {
        self.iteration_budget = Some(budget);
        self
    }
    /// Sets the preparation and workspace limits.
    pub const fn with_limits(mut self, limits: FftLogLimits) -> Self {
        self.limits = limits;
        self
    }
}
/// Reports the preparation work of an [`FftLogPlan`] and the spread of its multipliers.
#[derive(Debug, Clone, Copy)]
pub struct FftLogDiagnostics {
    /// Natural logarithm of the ratio of the largest to the smallest multiplier
    /// magnitude `|u_m|`; infinite if a multiplier is zero or a pole.
    pub log_multiplier_condition: f64,
    /// Work used during preparation, counted as in [`FftLogLimits::preparation_work`].
    pub preparation_work: usize,
}
/// Computes the fast Hankel transform on a logarithmic grid (FFTLog).
///
/// The forward transform approximates
///
/// ```text
/// ã(k) = ∫₀^∞ a(r) J_μ(kr) k dr,
/// ```
///
/// where `J_μ` is the Bessel function of the first kind of real order `μ`.
/// Input and output are sampled on increasing logarithmic grids of `n` points
/// with spacing `Δ`:
///
/// ```text
/// ln r_j = ln r_c + (j − (n−1)/2) Δ,   ln k_j = L − ln r_c + (j − (n−1)/2) Δ,
/// ```
///
/// so `ln r_j + ln k_(n−1−j) = L` for the offset `L`. The central radius `r_c`
/// is set by [`FftLogPlanSpec::with_log_radius_center`].
///
/// The method is Hamilton's. With bias `q`, the plan multiplies the input by
/// `(r_j/r_c)^(−q)`, extends it periodically in `ln r` (no padding or tapering
/// is applied), convolves it with the multipliers
///
/// ```text
/// u_m = 2^x Γ((μ + 1 + x)/2) / Γ((μ + 1 − x)/2) · e^(−i ω_m L),
/// x = q + i ω_m,   ω_m = 2π m / (n Δ),   m = −⌈n/2⌉ + 1, …, ⌊n/2⌋,
/// ```
///
/// and multiplies the result by `(k_j r_c)^(−q)`. For real `q` and even `n`,
/// the Nyquist multiplier is replaced by its real part. The inverse plan
/// applies `1/u_m` and the reciprocal power laws, so it inverts the forward
/// discrete map exactly in exact arithmetic. Through
/// [`AdjointOperator`](scioxide_core::AdjointOperator) a plan applies the
/// conjugate transpose of its own map.
///
/// A multiplier can be zero or a pole of the gamma ratio:
///
/// - If `(μ + 1 − x)/2` is a pole of `Γ`, then `u_m = 0`. A forward plan
///   removes that mode; an inverse plan returns `Singular` when executed.
/// - If `(μ + 1 + x)/2` is a pole of `Γ`, then `u_m` is a pole. A forward plan
///   returns `Singular` when executed; an inverse plan removes that mode.
/// - If both are poles, the ratio is evaluated by its finite limit.
///
/// The adjoint of a plan uses the same multipliers and fails in the same cases.
/// For real `q` and even `n`, a Nyquist multiplier whose real part is within
/// `64 ε_f64 |u_m|` of zero counts as zero.
///
/// Preparation evaluates `n` multipliers with the complex log-gamma function in
/// `f64`. Each execution costs two complex FFTs of length `n` and `O(n)` further
/// operations in the scalar type, and needs `n + A(n)` complex scratch
/// elements, where `A(n)` is the FFT's own scratch. Multipliers and power laws
/// are stored with separate binary exponents, so they may exceed the range of
/// the scalar type as long as the result does not. Each execution counts one
/// linear-operator evaluation and `n` iterations per transformed lane; see
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Examples
///
/// The inverse plan undoes the forward plan on the same grid.
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_fft::{FftDirection, FftLogPlan, FftLogPlanSpec};
///
/// let bias = Complex::new(0.2_f64, 0.0);
/// let spec = |direction| FftLogPlanSpec::new(5, 0.3, 0.5, 0.1, bias, direction);
/// let forward = FftLogPlan::new(spec(FftDirection::Forward))?;
/// let inverse = FftLogPlan::new(spec(FftDirection::Inverse))?;
/// let input = [1.0, -0.5, 2.0, 0.25, 0.75].map(|x| Complex::new(x, 0.0));
/// let (mut spectrum, mut recovered) = (input, input);
/// let cx = ExecutionContext::sequential_native();
/// forward.execute_into(&input, &mut spectrum, &mut forward.workspace(())?, cx)?;
/// inverse.execute_into(&spectrum, &mut recovered, &mut inverse.workspace(())?, cx)?;
/// for (y, x) in recovered.iter().zip(input) {
///     assert!((*y - x).norm() < 1e-10);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. J. S. Hamilton, "Uncorrelated modes of the non-linear power spectrum",
///   *Mon. Not. R. Astron. Soc.* 312, 257–284, 2000.
/// - J. D. Talman, "Numerical Fourier and Bessel transforms in logarithmic
///   variables", *J. Comput. Phys.* 29, 35–48, 1978.
#[derive(Debug, Clone)]
pub struct FftLogPlan<S: RealScalar> {
    plan_bytes: usize,
    spec: FftLogPlanSpec<S>,
    fft: ComplexCore<S>,
    /// Direction coefficients times `2^-coefficient_exponent`, bounded by one:
    /// Mellin multipliers forward and their reciprocals inverse.
    coefficients: Vec<Complex<S>>,
    coefficient_exponent: f64,
    /// Bias weights on the radial grid `x_j` and frequency grid `x_j + offset`.
    radial: BiasWeights<S>,
    frequency: BiasWeights<S>,
    /// Samples per weight block, shared by both grids.
    block: usize,
    scratch: usize,
    work: usize,
    condition: f64,
    headroom: i32,
    inverse_scale: S,
    /// Whether the direction's coefficients contain a zero (inverse) or a pole
    /// (forward); the adjoint shares the same coefficients.
    singular: bool,
}

/// Bias power law `exp(b (x_j + shift))` with separate binary exponents: factor
/// `j` times `2^exponents[j / block]`. Factors lie in `(2^-33, 1]` and each
/// exponent is an integer-valued `f64` of magnitude below 2^50, so
/// execution sums exponents exactly. A zero bias has unit weights and stores
/// no factors.
#[derive(Debug, Clone)]
struct BiasWeights<S: RealScalar> {
    factors: Option<Vec<Complex<S>>>,
    exponents: Vec<f64>,
}

/// Largest binary range spanned by one weight block.
const WEIGHT_BLOCK_RANGE: f64 = 32.0;
/// Exclusive magnitude bound of prepared binary exponents. Execution adds at
/// most three of them and scalar-range offsets, which stays exact in `f64`.
const EXPONENT_LIMIT: f64 = (1_u64 << 50) as f64;

#[derive(Clone, Copy)]
enum MellinCoefficient {
    Log(C),
    Zero,
    Pole,
}

// Scalar-width conversions report.
fn widen<S: RealScalar>(z: Complex<S>) -> Result<C, SciOxideError> {
    if !z.is_finite() {
        return Err(crate::validation::nonfinite());
    }
    Ok(C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?))
}
fn narrow<S: RealScalar>(z: C) -> Result<Complex<S>, SciOxideError> {
    // An overflowing complex exponential can also produce a NaN component
    // during multiplication. Preserve the arithmetic overflow classification.
    if z.re.is_infinite() || z.im.is_infinite() {
        return Err(crate::validation::arithmetic_overflow(
            "FFTLog result exceeds scalar range",
        ));
    }
    Ok(Complex::new(
        crate::validation::project(z.re)?,
        crate::validation::project(z.im)?,
    ))
}
// Gamma work is charged to `w`, whose operation reports every failure.
fn log_gamma(z: C, w: &mut Work, context: ExecutionContext<'_>) -> Result<C, SciOxideError> {
    w.step(context)?;
    let options = scioxide_special::GammaOptions::default()
        .with_iteration_budget(w.remaining_budget(context)?);
    // Fixed-cost log-gamma has no scoped form; its work joins the parent ledger here.
    let result = scioxide_special::ln_gamma_with_options(z, options);
    if let Err(error) = &result
        && let scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } =
            error.view()
    {
        let used = diagnostics.counts().total()?.iterations();
        w.charge_parent(used)?;
        w.record_special_iterations(used, false)?;
    }
    let r = result?;
    w.charge_parent(r.counts().iterations())?;
    w.record_special_iterations(
        r.counts().iterations(),
        r.termination() == scioxide_special::SpecialFunctionTermination::IterationLimit,
    )?;
    context.check()?;
    r.try_value()
}
fn multiplier_log(
    order: f64,
    bias: C,
    omega: f64,
    offset: f64,
    w: &mut Work,
    context: ExecutionContext<'_>,
) -> Result<C, SciOxideError> {
    // Hamilton's Mellin multiplier in log form for x = q + i*omega:
    //   ln u(x) = x ln 2 + ln Gamma((mu + 1 + x)/2) - ln Gamma((mu + 1 - x)/2)
    //             - i omega * offset,
    // evaluated with complex log-gamma so large |omega| never forms the
    // (over/underflowing) Gamma values themselves.
    let x = bias + C::new(0.0, omega);
    let a = (C::new(order + 1.0, 0.0) + x) * 0.5;
    let b = (C::new(order + 1.0, 0.0) - x) * 0.5;
    Ok(x * core::f64::consts::LN_2 + log_gamma(a, w, context)?
        - log_gamma(b, w, context)?
        - C::new(0.0, omega * offset))
}
// At real gamma poles the ratio is evaluated by its meromorphic limit
// with fixed bias (the Pochhammer-symbol limit).
fn mellin_coefficient(
    order: f64,
    bias: C,
    omega: f64,
    offset: f64,
    work: &mut Work,
    context: ExecutionContext<'_>,
) -> Result<MellinCoefficient, SciOxideError> {
    let x = bias + C::new(0.0, omega);
    let a = (C::new(order + 1.0, 0.0) + x) * 0.5;
    let b = (C::new(order + 1.0, 0.0) - x) * 0.5;
    let pole = |z: C| z.im == 0.0 && z.re <= 0.0 && z.re == z.re.floor();
    match (pole(a), pole(b)) {
        (false, true) => {
            work.step(context)?;
            Ok(MellinCoefficient::Zero)
        }
        (true, false) => {
            work.step(context)?;
            Ok(MellinCoefficient::Pole)
        }
        (true, true) => {
            let sign = if (a.re - b.re) % 2.0 == 0.0 {
                0.0
            } else {
                core::f64::consts::PI
            };
            let value = x * core::f64::consts::LN_2
                + log_gamma(C::new(1.0 - b.re, 0.0), work, context)?
                - log_gamma(C::new(1.0 - a.re, 0.0), work, context)?
                + C::new(0.0, sign - omega * offset);
            Ok(MellinCoefficient::Log(value))
        }
        (false, false) => {
            multiplier_log(order, bias, omega, offset, work, context).map(MellinCoefficient::Log)
        }
    }
}
// Prepared binary exponents must stay exact when execution sums them.
fn exact_exponent(exponent: f64) -> Result<f64, SciOxideError> {
    if exponent.abs() < EXPONENT_LIMIT {
        Ok(exponent)
    } else {
        Err(crate::validation::arithmetic_overflow(
            "FFTLog binary exponent exceeds the exact binary64 range",
        ))
    }
}
// Saturating beyond every scalar's range lets `BinaryScale` round to zero or
// infinity exactly as `scalbn` does.
fn binary_exponent(exponent: f64) -> i32 {
    exponent.clamp(-65536.0, 65536.0) as i32
}
// Samples per weight block, so that one block spans at most
// `WEIGHT_BLOCK_RANGE` binary orders of the bias power law.
fn weight_block(n: usize, bias: C, spacing: f64) -> usize {
    let length = (WEIGHT_BLOCK_RANGE / (bias.re.abs() * spacing / LN_2)).floor() + 1.0;
    if length < n as f64 {
        length as usize
    } else {
        n
    }
}
impl<S: RealScalar> BiasWeights<S> {
    /// Tabulate `exp(bias * (x_j + shift))` for `x_j = (j - (n-1)/2) spacing`.
    fn new(
        bias: C,
        spacing: f64,
        shift: f64,
        n: usize,
        block: usize,
    ) -> Result<Self, SciOxideError> {
        let center = (n - 1) as f64 * 0.5;
        let log = |j: usize| {
            let coordinate = (j as f64 - center) * spacing + shift;
            // A zero bias component contributes nothing, even where an
            // extreme grid coordinate overflows.
            let product = |component: f64| {
                if component == 0.0 {
                    0.0
                } else {
                    component * coordinate
                }
            };
            let value = C::new(product(bias.re), product(bias.im));
            if value.is_finite() {
                Ok(value)
            } else {
                Err(crate::validation::arithmetic_overflow(
                    "FFTLog bias weight exceeds binary64 range",
                ))
            }
        };
        let mut exponents = try_filled(
            n.div_ceil(block),
            0.0,
            scioxide_core::quantity!("fftlog_bias_exponents"),
        )?;
        if bias == C::new(0.0, 0.0) {
            return Ok(Self {
                factors: None,
                exponents,
            });
        }
        let mut factors = try_filled(
            n,
            Complex::new(S::ZERO, S::ZERO),
            scioxide_core::quantity!("fftlog_bias_weights"),
        )?;
        for ((index, chunk), exponent) in factors.chunks_mut(block).enumerate().zip(&mut exponents)
        {
            let samples = index * block..index * block + chunk.len();
            let largest = samples.clone().try_fold(f64::NEG_INFINITY, |largest, j| {
                log(j).map(|w| largest.max(w.re))
            })?;
            *exponent = exact_exponent((largest / LN_2).ceil())?;
            for (factor, j) in chunk.iter_mut().zip(samples) {
                let w = log(j)?;
                *factor = narrow::<S>(C::new(w.re - *exponent * LN_2, w.im).exp())?;
            }
        }
        Ok(Self {
            factors: Some(factors),
            exponents,
        })
    }
    /// Each block's factors, absent for unit weights, and binary exponent.
    fn blocks(&self, block: usize) -> impl Iterator<Item = (Option<&[Complex<S>]>, f64)> {
        let mut factors = self.factors.as_deref().map(|factors| factors.chunks(block));
        self.exponents
            .iter()
            .map(move |&exponent| (factors.as_mut().and_then(Iterator::next), exponent))
    }
    fn bytes(&self) -> [usize; 2] {
        [
            self.factors.as_ref().map_or(0, crate::memory::vector_bytes),
            crate::memory::vector_bytes(&self.exponents),
        ]
    }
}
impl<S: RealScalar> FftLogPlan<S> {
    /// Returns the bytes held by the plan, including its FFT, excluding allocator metadata.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("fft_plan_bytes");
        [
            crate::memory::vector_bytes(&self.coefficients),
            self.fft.plan_bytes() - size_of::<ComplexCore<S>>(),
        ]
        .into_iter()
        .chain(self.radial.bytes())
        .chain(self.frequency.bytes())
        .try_fold(size_of::<Self>(), |a, b| checked_add(a, b, q))
    }

    /// Prepares a plan with the default FFT planning context on the calling thread.
    ///
    /// A spec whose multipliers contain a zero or a pole still prepares; see the
    /// type documentation for which executions then return `Singular`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the length is zero or `log_spacing ≤ 0`.
    /// - `NonFinite` if the order, spacing, offset, center or bias is NaN or
    ///   infinite.
    /// - `NumericalInstability` (`Underflow`) if a nonzero multiplier rounds to
    ///   zero in the scalar type, or (`Overflow`) if a power-law weight or a
    ///   binary exponent is out of range.
    /// - `ResourceLimit` if preparation or workspace storage exceeds
    ///   [`FftLogLimits`], and `ConvergenceFailure` if the preparation work is
    ///   exhausted.
    pub fn new(spec: FftLogPlanSpec<S>) -> Result<Self, SciOxideError> {
        Self::build(
            spec,
            FftPlanningContext::default(),
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan with an explicit FFT planning context and solve scope.
    ///
    /// `planning` selects the FFT backend or provider (see
    /// [Backends and providers](crate#backends-and-providers)). Preparation work
    /// counts toward `scope` as described in
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], plus capability, provider, cancellation and
    /// deadline errors from `planning` and `scope`.
    pub fn build<'p, 'scope, 'cap>(
        spec: FftLogPlanSpec<S>,
        planning: impl Into<FftPlanningContext<'p, S>>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        let planning = planning.into();
        let scope = scope.into();

        scope.execution().check()?;
        let mut state = scioxide_core::SolveState::new();
        scope.run(
            &mut state,
            crate::workspace::identity(spec.length),
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Iteration,
                spec.limits.preparation_work,
            ),
            |child| {
                let plan = Self::prepare(spec, planning, child)?;
                child.execution().check()?;
                Ok(plan)
            },
        )
    }

    // Preparation phases preserve validation and numerical error precedence.
    fn prepare(
        spec: FftLogPlanSpec<S>,
        context: FftPlanningContext<'_, S>,
        parent: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let request = Self::prepare_request(&spec)?;
        let context = context.with_provider_limits(crate::FftProviderLimits {
            preparation_bytes: spec.limits.preparation_bytes - request.outer_bytes,
            workspace_bytes: spec
                .limits
                .workspace_bytes
                .saturating_sub(request.staging_bytes),
        });
        let mut work = Work::new(spec.limits.preparation_work, parent);
        let PreparedFft { fft, scratch } = Self::prepare_fft(&spec, &request, context, &mut work)?;
        let spectrum = Self::prepare_spectrum(&request, &mut work, context.execution())?;
        let forward = matches!(spec.direction, crate::FftDirection::Forward);
        let PreparedCoefficients {
            values: coefficients,
            exponent: coefficient_exponent,
            condition,
            singular,
        } = Self::prepare_coefficients(spectrum, forward)?;
        let PreparedWeights { radial, frequency } = Self::prepare_weights(&request, forward)?;
        work.check(context.execution())?;
        let n = spec.length;
        let mut plan = Self {
            plan_bytes: 0,
            spec,
            fft,
            coefficients,
            coefficient_exponent,
            radial,
            frequency,
            block: request.block,
            scratch,
            work: work.used,
            condition,
            singular,
            // Staged weights and coefficients are bounded by one; the first FFT grows by at most n.
            headroom: crate::scaling::growth_exponent(n) + 3,
            inverse_scale: crate::numeric::scalar_from_usize_rounded::<S>(n)?.recip(),
        };
        plan.plan_bytes = plan.retained_bytes()?;
        Ok(plan)
    }

    fn prepare_request(spec: &FftLogPlanSpec<S>) -> Result<PreparationRequest, SciOxideError> {
        let n = spec.length;
        if !spec.order.is_finite()
            || !spec.log_spacing.is_finite()
            || !spec.offset.is_finite()
            || !spec.log_radius_center.is_finite()
        {
            return Err(crate::validation::nonfinite());
        }
        crate::validation::nonzero(n, scioxide_core::argument!("spec.length"))?;
        if spec.log_spacing <= S::ZERO {
            return Err(invalid(
                scioxide_core::InvalidInputReason::OutOfRange,
                scioxide_core::argument!("spec.log_spacing"),
                "positive FFTLog spacing required",
            ));
        }
        // Gamma coefficients and bias weights are prepared in binary64; FFTs,
        // normalized tables and transactional execution storage use S.
        let count = f64::try_from_usize_exact(n)?;
        let bias = widen(spec.bias)?;
        let spacing = spec.log_spacing.try_to_f64_exact()?;
        let offset = spec.offset.try_to_f64_exact()?;
        let order = spec.order.try_to_f64_exact()?;
        let block = weight_block(n, bias, spacing);
        let qbytes = scioxide_core::quantity!("fftlog_preparation_bytes");
        // Gamma logs, coefficients and both weight tables bound peak storage;
        // a zero bias stores no weight factors.
        let tables = if bias == C::new(0.0, 0.0) { 1 } else { 3 };
        let outer_bytes = [
            checked_mul(
                n,
                size_of::<MellinCoefficient>() + tables * size_of::<Complex<S>>(),
                qbytes,
            )?,
            checked_mul(n.div_ceil(block), 2 * size_of::<f64>(), qbytes)?,
        ]
        .into_iter()
        .try_fold(size_of::<Self>() - size_of::<ComplexCore<S>>(), |a, b| {
            checked_add(a, b, qbytes)
        })?;
        admit_bytes(outer_bytes, spec.limits.preparation_bytes, qbytes)?;
        let staging_bytes = checked_mul(
            n,
            size_of::<Complex<S>>(),
            scioxide_core::quantity!("fftlog_workspace_bytes"),
        )?;
        Ok(PreparationRequest {
            n,
            count,
            bias,
            spacing,
            offset,
            order,
            block,
            outer_bytes,
            staging_bytes,
        })
    }

    fn prepare_fft(
        spec: &FftLogPlanSpec<S>,
        request: &PreparationRequest,
        context: FftPlanningContext<'_, S>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<PreparedFft<S>, SciOxideError> {
        let n = spec.length;
        work.step(context.execution())?;
        let fft = ComplexCore::with_preparation_limit(
            n,
            spec.execution_policy,
            spec.backend_strategy,
            context,
            scioxide_core::argument!("spec.length"),
            scioxide_core::argument!("planning"),
            spec.limits.preparation_bytes - request.outer_bytes,
        )?;
        let scratch = checked_add(
            n,
            fft.bounded_scratch_len(),
            scioxide_core::quantity!("fftlog_scratch"),
        )?;
        admit_bytes(
            checked_mul(
                scratch,
                size_of::<Complex<S>>(),
                scioxide_core::quantity!("fftlog_scratch_bytes"),
            )?,
            spec.limits.workspace_bytes,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        Ok(PreparedFft { fft, scratch })
    }

    fn prepare_spectrum(
        request: &PreparationRequest,
        work: &mut Work<'_, '_, '_>,
        context: ExecutionContext<'_>,
    ) -> Result<MellinSpectrum, SciOxideError> {
        let PreparationRequest {
            n,
            count,
            bias,
            spacing,
            offset,
            order,
            ..
        } = *request;
        let mut logs = try_filled(
            n,
            MellinCoefficient::Zero,
            scioxide_core::quantity!("fftlog_log_multipliers"),
        )?;
        let (mut maximum, mut minimum) = (f64::NEG_INFINITY, f64::INFINITY);
        let (mut forward_singular, mut inverse_singular) = (false, false);
        for (k, out) in logs.iter_mut().enumerate() {
            let frequency = if k <= n / 2 {
                k as f64
            } else {
                k as f64 - count
            };
            let omega = 2.0 * core::f64::consts::PI * frequency / (count * spacing);
            *out = mellin_coefficient(order, bias, omega, offset, work, context)?;
            if let MellinCoefficient::Log(log) = out
                && n.is_multiple_of(2)
                && k == n / 2
                && bias.im == 0.0
            {
                // The real interpolant projects its single Nyquist bin to
                // the mean of conjugate limits. A zero only removes a mode
                // from forward execution; it makes the inverse singular.
                let cosine = log.im.cos();
                if cosine.abs() <= 64.0 * f64::EPSILON {
                    *out = MellinCoefficient::Zero;
                } else {
                    log.re += cosine.abs().ln();
                    log.im = if cosine < 0.0 {
                        core::f64::consts::PI
                    } else {
                        0.0
                    };
                }
            }
            match *out {
                MellinCoefficient::Zero => inverse_singular = true,
                MellinCoefficient::Pole => forward_singular = true,
                MellinCoefficient::Log(log) => {
                    narrow::<f64>(log)?;
                    maximum = maximum.max(log.re);
                    minimum = minimum.min(log.re);
                }
            }
        }
        Ok(MellinSpectrum {
            logs,
            maximum,
            minimum,
            forward_singular,
            inverse_singular,
        })
    }

    fn prepare_coefficients(
        spectrum: MellinSpectrum,
        forward: bool,
    ) -> Result<PreparedCoefficients<S>, SciOxideError> {
        let MellinSpectrum {
            logs,
            maximum,
            minimum,
            forward_singular,
            inverse_singular,
        } = spectrum;
        let n = logs.len();
        let condition = if forward_singular || inverse_singular {
            f64::INFINITY
        } else {
            maximum - minimum
        };
        // A plan and its adjoint use only the direction's coefficients:
        // multipliers u forward (a pole is singular, a zero annihilates its
        // mode) and reciprocals 1/u inverse (the roles swap).
        let (singular, sign, largest) = if forward {
            (forward_singular, 1.0, maximum)
        } else {
            (inverse_singular, -1.0, -minimum)
        };
        // A power-of-two normalization bounds every coefficient by one and
        // leaves an exact binary exponent; a one-point transform may consist
        // solely of a zero/pole.
        let coefficient_exponent = if largest.is_finite() {
            (largest / LN_2).ceil()
        } else {
            0.0
        };
        let zero = Complex::new(S::ZERO, S::ZERO);
        let mut coefficients =
            try_filled(n, zero, scioxide_core::quantity!("fftlog_coefficients"))?;
        if !singular {
            for (coefficient, log) in coefficients.iter_mut().zip(&logs) {
                if let MellinCoefficient::Log(log) = *log {
                    *coefficient =
                        narrow::<S>((log * sign - C::new(coefficient_exponent * LN_2, 0.0)).exp())?;
                    if *coefficient == zero {
                        return Err(numerical(
                            scioxide_core::NumericalIssue::Underflow,
                            "FFTLog coefficient dynamic range exceeds scalar precision",
                        ));
                    }
                }
            }
        }
        drop(logs);
        let coefficient_exponent = exact_exponent(coefficient_exponent)?;
        Ok(PreparedCoefficients {
            values: coefficients,
            exponent: coefficient_exponent,
            condition,
            singular,
        })
    }

    fn prepare_weights(
        request: &PreparationRequest,
        forward: bool,
    ) -> Result<PreparedWeights<S>, SciOxideError> {
        let PreparationRequest {
            n,
            bias,
            spacing,
            offset,
            block,
            ..
        } = *request;
        // Inverse weights undo forward weights; the adjoint conjugates them.
        let bias = if forward { -bias } else { bias };
        let radial = BiasWeights::new(bias, spacing, 0.0, n, block)?;
        let frequency = BiasWeights::new(bias, spacing, offset, n, block)?;
        Ok(PreparedWeights { radial, frequency })
    }
    /// Returns the spec the plan was prepared from.
    pub const fn spec(&self) -> &FftLogPlanSpec<S> {
        &self.spec
    }
    /// Returns the preparation work and the spread of the multipliers.
    pub const fn diagnostics(&self) -> FftLogDiagnostics {
        FftLogDiagnostics {
            log_multiplier_condition: self.condition,
            preparation_work: self.work,
        }
    }
    /// Returns the backend of the plan's FFT.
    pub fn resolved_backends(&self) -> impl Iterator<Item = FftBackend> + '_ {
        core::iter::once(self.fft.resolved_backend())
    }
    /// Returns the workspace size for [`Self::execute_into`] (request `()`) or for
    /// [`Self::execute_axis_into`] (an [`AxisWorkspaceRequest`](crate::AxisWorkspaceRequest)).
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` or `DimensionMismatch` if an axis request does not match
    ///   the plan.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if it exceeds [`FftLogLimits::workspace_bytes`].
    pub fn workspace_requirements<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        crate::axis::requirements(self, request.into())
    }
    /// Allocates a workspace for the given request.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }

    /// Returns `ln r_j`, the natural logarithm of input radius `j`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `j` is not below the length.
    /// - `NumericalInstability` (`Overflow`) if the coordinate overflows.
    pub fn log_radius(&self, j: usize) -> Result<S, SciOxideError> {
        self.coordinate(j, false)
    }
    /// Returns `ln k_j`, the natural logarithm of output frequency `j`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `j` is not below the length.
    /// - `NumericalInstability` (`Overflow`) if the coordinate overflows.
    pub fn log_frequency(&self, j: usize) -> Result<S, SciOxideError> {
        self.coordinate(j, true)
    }
    fn coordinate(&self, j: usize, frequency: bool) -> Result<S, SciOxideError> {
        if j >= self.spec.length {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("fftlog_samples"),
                j,
                self.spec.length,
            ));
        }
        let center = if frequency {
            self.spec.offset - self.spec.log_radius_center
        } else {
            self.spec.log_radius_center
        };
        let value = center
            + S::try_from_f64_lossy(j as f64 - (self.spec.length - 1) as f64 * 0.5)?
                * self.spec.log_spacing;
        if !value.is_finite() {
            return Err(crate::validation::arithmetic_overflow(
                "log-grid coordinate overflow",
            ));
        }
        Ok(value)
    }

    /// Transforms `input` in the plan's direction and writes the result to `output`.
    ///
    /// Both slices have the plan length. `scope` is an execution context or a
    /// parent solve context; the call counts one linear-operator evaluation and
    /// `n` iterations. Components that underflow are rounded, not reported.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice has the wrong length or the workspace is
    ///   too small.
    /// - `Singular` if the plan's direction has a pole (forward) or a zero
    ///   (inverse) multiplier.
    /// - `NumericalInstability` (`Overflow`) if a result exceeds the scalar range.
    /// - `ConvergenceFailure` if the iteration or evaluation budget is exhausted.
    /// - `NonFinite`, `ExecutionStopped` and provider errors as described in the
    ///   [crate errors](crate#errors).
    pub fn execute_into<'scope, 'cap>(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute_radial(self, input, output, workspace, scope.into())
    }

    /// Transforms every lane of `input` along `axis` and writes the results to `output`.
    ///
    /// The other axes are batch axes. Input views may repeat elements (stride
    /// zero); output views may not. Either every lane is written or, on error,
    /// none is. The call counts one linear-operator evaluation if there is at
    /// least one lane.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::execute_into`], plus `OutOfBounds` for an invalid
    /// axis and `DimensionMismatch` for shapes that do not match the plan.
    pub fn execute_axis_into<'scope, 'cap, T: scioxide_core::NumericScalar<Real = S>>(
        &self,
        input: scioxide_core::NdView<'_, '_, T>,
        axis: usize,
        output: scioxide_core::NdViewMut<'_, '_, Complex<S>>,
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute(self, input, axis, output, workspace, scope.into())
    }
    // Shared transactional execution kernel; the result stays in `scratch`.
    fn evaluate<'w>(
        &self,
        input: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        adjoint: bool,
        parent: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        let context = parent.execution();
        let n = self.spec.length;
        length(input.len(), n)?;
        crate::validation::sufficient(scratch.len(), self.scratch)?;
        // The log-frequency grid runs opposite to the log-radius grid
        // (ln k_j = offset - ln r_{n-1-j}), so the Mellin convolution output
        // is index-reversed. Forward and inverse-adjoint execution weight the
        // radial grid first and reverse the output; the inverse and adjoint
        // (their transposes) weight the frequency grid first and reverse the
        // input.
        let radial_first = matches!(self.spec.direction, crate::FftDirection::Forward) != adjoint;
        let (pre, post) = if radial_first {
            (&self.radial, &self.frequency)
        } else {
            (&self.frequency, &self.radial)
        };
        // Log-domain protection: `common` is the largest staged binary
        // exponent, so every block scaled by 2^(exponent - common) stays
        // bounded even when the bias power law alone would overflow or
        // underflow. The scan also validates the input.
        let common = input.chunks(self.block).zip(&pre.exponents).try_fold(
            f64::NEG_INFINITY,
            |common, (values, &exponent)| {
                let largest = complex_max(values.iter().copied())?;
                Ok::<_, SciOxideError>(if largest == S::ZERO {
                    common
                } else {
                    common.max(f64::from(range_exponent(largest)) + exponent)
                })
            },
        )?;
        context.check()?;
        if self.singular {
            return Err(SciOxideError::singular(None));
        }
        crate::resource::admit_iterations(parent, n)?;
        let (stage, fft_work) = scratch.split_at_mut(n);
        if common == f64::NEG_INFINITY {
            context.check()?;
            stage.fill(Complex::new(S::ZERO, S::ZERO));
            return Ok(stage);
        }
        // The adjoint conjugates weight factors and coefficients.
        let sign = if adjoint { -S::ONE } else { S::ONE };
        let oriented = |factor: Complex<S>| Complex::new(factor.re, factor.im * sign);
        let headroom = f64::from(self.headroom);
        for ((staged, values), (factors, exponent)) in stage
            .chunks_mut(self.block)
            .zip(input.chunks(self.block))
            .zip(pre.blocks(self.block))
        {
            // Scaling before weighting keeps unit factors from overflowing;
            // components that underflow are negligible beside the maximum.
            let scale = BinaryScale::new(binary_exponent(exponent - common - headroom));
            let scaled =
                |value: Complex<S>| Complex::new(scale.apply(value.re), scale.apply(value.im));
            match factors {
                None => {
                    for (staged, &value) in staged.iter_mut().zip(values) {
                        *staged = scaled(value);
                    }
                }
                Some(factors) => {
                    for ((staged, &value), &factor) in staged.iter_mut().zip(values).zip(factors) {
                        *staged = scaled(value) * oriented(factor);
                    }
                }
            }
        }
        if !radial_first {
            stage.reverse();
        }
        self.fft
            .transform_bounded(stage, fft_work, false, S::ONE, context)?;
        for (value, &coefficient) in stage.iter_mut().zip(&self.coefficients) {
            *value *= oriented(coefficient);
        }
        self.fft
            .transform_bounded(stage, fft_work, true, self.inverse_scale, context)?;
        if radial_first {
            stage.reverse();
        }
        // Weighting before restoration leaves the final binary scale as the
        // only rounding that can overflow or reach the subnormal range.
        let lift = common + headroom + self.coefficient_exponent;
        let mut finite = true;
        for (values, (factors, exponent)) in
            stage.chunks_mut(self.block).zip(post.blocks(self.block))
        {
            let scale = BinaryScale::new(binary_exponent(exponent + lift));
            let scaled =
                |value: Complex<S>| Complex::new(scale.apply(value.re), scale.apply(value.im));
            match factors {
                None => {
                    for value in values.iter_mut() {
                        *value = scaled(*value);
                        finite &= value.re.is_finite() & value.im.is_finite();
                    }
                }
                Some(factors) => {
                    for (value, &factor) in values.iter_mut().zip(factors) {
                        *value = scaled(*value * oriented(factor));
                        finite &= value.re.is_finite() & value.im.is_finite();
                    }
                }
            }
        }
        if !finite {
            return Err(crate::validation::arithmetic_overflow(
                "FFTLog result exceeds scalar range",
            ));
        }
        context.check()?;
        Ok(stage)
    }
}
impl<S: RealScalar> AxisPlan<S> for FftLogPlan<S> {
    fn lane_lens(&self) -> (usize, usize) {
        (self.spec.length, self.spec.length)
    }
    fn scratch(&self) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        let scratch = crate::FftWorkspaceRequirements::new::<S>(self.scratch)?;
        scratch.with_limit(crate::workspace::nested_byte_limit(
            self.spec.limits.workspace_bytes,
            scratch.bytes(),
            [(
                self.fft.bounded_workspace_byte_limit(),
                self.fft.bounded_scratch_len() * size_of::<Complex<S>>(),
            )],
        ))
    }
    fn workspace_bytes(&self) -> usize {
        self.spec.limits.workspace_bytes
    }
    fn iteration_budget(&self) -> Option<scioxide_core::Budget> {
        self.spec.iteration_budget
    }
    fn admit(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        crate::resource::admit_operator(context)
    }
    fn lane<'w>(
        &self,
        source: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(source, scratch, false, context)
    }
}
/// Returns the low-ringing FFTLog offset nearest to `initial` for a real bias.
///
/// The parameters are those of [`FftLogPlanSpec::new`]: spacing `Δ`, order `μ`
/// and bias `q`. Hamilton's low-ringing condition asks the Nyquist multiplier
/// `u_(n/2)` of [`FftLogPlan`], at `ω = π/Δ`, to be real. Its phase decreases
/// by `π` each time the offset `L` grows by `Δ`, so the function returns the
/// offset with an integer phase multiple of `π` that is nearest to `initial`;
/// it differs from `initial` by at most `Δ/2`.
///
/// # Errors
///
/// - `NonFinite` if an argument is NaN or infinite.
/// - `InvalidInput` if `log_spacing ≤ 0`.
/// - An error from the complex log-gamma function if `(μ + 1 ± q ± iπ/Δ)/2` is
///   a pole of `Γ`, or if it needs more than 65,536 work units.
///
/// # Examples
///
/// ```
/// // Reference value from an independent implementation of Hamilton's formula.
/// let offset = scioxide_fft::fftlog_low_ringing_offset(0.1_f64, 2.0, 0.5, 0.0)?;
/// assert!((offset - 0.5454581477676637).abs() < 2e-14);
/// assert!((offset - 0.5).abs() <= 0.1 / 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. J. S. Hamilton, "Uncorrelated modes of the non-linear power spectrum",
/// *Mon. Not. R. Astron. Soc.* 312, 257–284, 2000.
pub fn fftlog_low_ringing_offset<S: RealScalar>(
    log_spacing: S,
    order: S,
    initial: S,
    bias: S,
) -> Result<S, SciOxideError> {
    if [log_spacing, order, initial, bias]
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err(crate::validation::nonfinite());
    }
    let (spacing, order, initial, bias) = (
        log_spacing.try_to_f64_exact()?,
        order.try_to_f64_exact()?,
        initial.try_to_f64_exact()?,
        bias.try_to_f64_exact()?,
    );
    if spacing <= 0.0 {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("log_spacing"),
            "log spacing must be positive",
        ));
    }
    // Hamilton's low-ringing choice makes the Nyquist multiplier real: its
    // phase contains -omega * offset with omega = pi / spacing, so moving the
    // offset by (phase/pi - round(phase/pi)) * spacing turns phase/pi into
    // the nearest integer.
    let omega = core::f64::consts::PI / spacing;
    let execution = ExecutionContext::sequential();
    let mut state = scioxide_core::SolveState::new();
    scioxide_core::SolveScope::from(execution).run(
        &mut state,
        crate::workspace::identity(1),
        scioxide_core::SolveBudgets::unbounded(),
        |child| {
            let mut work = Work::new(scioxide_core::Budget::ONE.saturating_add(65535), child);
            let phase = multiplier_log(
                order,
                C::new(bias, 0.0),
                omega,
                initial,
                &mut work,
                execution,
            )?
            .im / core::f64::consts::PI;
            S::try_from_f64_lossy(initial + (phase - phase.round()) * spacing)
        },
    )
}
impl<S: RealScalar> crate::axis::RadialPlan<S> for FftLogPlan<S> {
    fn evaluate<'w>(
        &self,
        input: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        adjoint: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(input, scratch, adjoint, context)
    }
}
crate::axis::impl_radial_operators!(FftLogPlan);

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::AdjointOperator;

    #[test]
    fn singular_real_nyquist_admits_forward_adjoint_and_rejects_inverse() {
        let d = 0.125;
        let omega = core::f64::consts::PI / d;
        let context = ExecutionContext::sequential();
        let mut state = scioxide_core::SolveState::new();
        let phase = scioxide_core::SolveScope::from(context)
            .run(
                &mut state,
                crate::workspace::identity(8),
                scioxide_core::SolveBudgets::unbounded(),
                |child| {
                    let mut work =
                        Work::new(scioxide_core::Budget::try_from(65536).unwrap(), child);
                    Ok(multiplier_log(0.0, C::new(0.0, 0.0), omega, 0.0, &mut work, context)?.im)
                },
            )
            .unwrap();
        let offset = (phase - core::f64::consts::FRAC_PI_2) / omega;
        let plan = FftLogPlan::new(FftLogPlanSpec::new(
            8,
            d,
            0.0,
            offset,
            C::new(0.0, 0.0),
            crate::FftDirection::Forward,
        ))
        .unwrap();
        let plan_inverse = FftLogPlan::new(FftLogPlanSpec::new(
            8,
            d,
            0.0,
            offset,
            C::new(0.0, 0.0),
            crate::FftDirection::Inverse,
        ))
        .unwrap();
        let input: Vec<_> = (0..8)
            .map(|j| C::new(if j % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();
        let mut output = input.clone();
        let mut scratch = plan.workspace(()).unwrap();
        plan.execute_into(&input, &mut output, &mut scratch, context)
            .unwrap();
        assert!(output.iter().all(|z| z.norm() < 1e-14));
        let expected = output.clone();
        assert!(matches!(
            plan_inverse
                .execute_into(&input, &mut output, &mut scratch, context)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
        assert_eq!(output, expected);
        assert!(matches!(
            plan_inverse
                .apply_adjoint_into(&input, &mut output, &mut scratch)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
        assert_eq!(output, expected);
        // The forward adjoint conjugates the same zero coefficient, so it also
        // annihilates the (reversed) Nyquist mode.
        plan.apply_adjoint_into(&input, &mut output, &mut scratch)
            .unwrap();
        assert!(output.iter().all(|z| z.norm() < 1e-14));
    }
}
