//! Kolmogorov–Smirnov laws: the limiting Kolmogorov law, exact one-sided sums and Durbin matrices.
use super::*;
use crate::distribution::LawOptions;
use alloc::vec::Vec;
use math::{boundary, integral};
use scioxide_core::Capacity;

const KOLMOGOROV_MEDIAN: f64 = 0.827_573_555_189_907_7;

/// Kolmogorov distribution: the limiting law of `√n Dₙ` for the two-sided Kolmogorov–Smirnov
/// statistic `Dₙ`.
///
/// The distribution function is
///
/// ```text
/// K(x) = 1 − 2 Σ_{k≥1} (−1)^(k−1) e^(−2k²x²) = (√(2π)/x) Σ_{k≥1} e^(−(2k−1)²π²/(8x²)),   x > 0.
/// ```
///
/// The first series is used for large `x` and the second (Jacobi theta) form
/// for small `x`. Each computes the tail that is small in its range directly.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Kolmogorov, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Kolmogorov::new()?;
/// let median = 0.827_573_555_189_907_7_f64;
/// assert!((law.cdf(median)? - 0.5).abs() < 1e-12 && (law.sf(median)? - 0.5).abs() < 1e-12);
/// assert!((law.quantile(0.5)? - median).abs() < 1e-10);
/// let mean = (core::f64::consts::PI / 2.0).sqrt() * core::f64::consts::LN_2;
/// assert!((law.mean()? - mean).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Kolmogorov<S: RealScalar> {
    options: LawOptions<S>,
}
impl<S: RealScalar> Kolmogorov<S> {
    /// Creates the Kolmogorov law.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    pub fn new() -> Result<Self, SciOxideError> {
        Self::with_options(LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self { options })
    }
}
impl<S: RealScalar> Statistic<S> for Kolmogorov<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn median(&self) -> f64 {
        KOLMOGOROV_MEDIAN
    }
    fn ordinary(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let y = x.into_f64();
        let exponent = if y < 0.82 {
            -core::f64::consts::PI.powi(2) / (8.0 * y * y)
        } else {
            -2.0 * y * y
        };
        if !exponent.is_finite() {
            return Ok(if (y < 0.82) == upper { S::ONE } else { S::ZERO });
        }
        let mut work = Work::new(self.options);
        number(self.tail_parts(y, upper, &mut work)?.0.exp())
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
            return Ok(v);
        }
        // Two classical series for the Kolmogorov law. For small x the Jacobi
        // theta form F(x) = sqrt(2 pi)/x sum exp(-(2k-1)^2 pi^2 / (8x^2))
        // converges fast; for large x the alternating
        // S(x) = 2 sum (-1)^(k-1) exp(-2 k^2 x^2) does. 0.82 is roughly where
        // both need a similar number of terms. The leading exponential is
        // factored out so each sum starts at 1 and stays representable, and the
        // series computes whichever tail is small in its regime directly.
        let small = x < 0.82;
        let leading = if small {
            -core::f64::consts::PI.powi(2) / (8.0 * x * x)
        } else {
            -2.0 * x * x
        };
        if !leading.is_finite() {
            if q != Query::Density && (q == Query::Lower) != small {
                return Ok(0.0);
            }
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "Kolmogorov logarithm exceeds the finite scalar range",
            ));
        }
        let mut sum = 0.0;
        let mut derivative = 0.0;
        for k in 1usize.. {
            w.charge(1)?;
            let k = k as f64;
            let power = if small {
                (2.0 * k - 1.0).powi(2)
            } else {
                k * k
            };
            let t = (leading * (power - 1.0)).exp();
            let sign = if small || k as usize % 2 == 1 {
                1.0
            } else {
                -1.0
            };
            sum += sign * t;
            derivative += sign
                * t
                * if small {
                    (-1.0 - 2.0 * leading * power) / x
                } else {
                    4.0 * x * power
                };
            if k > 1.0 && t < w.tolerance * sum.abs() * 0.01 {
                break;
            }
        }
        let prefactor = if small {
            0.5 * crate::distribution::special::log_two_pi::<f64>() - x.ln()
        } else {
            core::f64::consts::LN_2
        };
        if q == Query::Density {
            return Ok(prefactor + leading + derivative.ln());
        }
        let value = prefactor + leading + sum.ln();
        Ok(if (q == Query::Lower) == small {
            value
        } else {
            log1mexp(value)?
        })
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.quantile(crate::float::open_word_probability(
            rng,
            self.options.sample_words,
        )?)
    }
}
statistic_law!(Kolmogorov, report);
impl<S: RealScalar> Mean<S> for Kolmogorov<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number((core::f64::consts::PI / 2.0).sqrt() * core::f64::consts::LN_2)
    }
}
impl<S: RealScalar> Variance<S> for Kolmogorov<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        number(
            core::f64::consts::PI.powi(2) / 12.0
                - core::f64::consts::PI / 2.0 * core::f64::consts::LN_2.powi(2),
        )
    }
}

fn count(n: usize) -> Result<f64, SciOxideError> {
    if n == 0 {
        return Err(invalid(
            scioxide_core::argument!("n"),
            "KS sample count must be positive",
        ));
    }
    f64::try_from_usize_exact(n)
}
/// Exact null distribution of the one-sided Kolmogorov–Smirnov statistic `Dₙ⁺` for a sample of size
/// `n`.
///
/// `Dₙ⁺ = sup_x (Fₙ(x) − F(x))` for the empirical distribution function `Fₙ`
/// of `n` independent draws from a continuous `F`. The support is `[0, 1]`.
/// For `x ≤ 1/n` the distribution function is `x (1 + x)^(n−1)`. Otherwise
/// the upper tail is the Birnbaum–Tingey sum
///
/// ```text
/// P(Dₙ⁺ ≥ x) = x Σ_{j=0}^{⌊n(1−x)⌋} C(n, j) (x + j/n)^(j−1) (1 − x − j/n)^(n−j)
/// ```
///
/// which takes O(n) time.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, KolmogorovSmirnovOneSided, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64};
/// // For x <= 1/n, P(D+ <= x) = x (1 + x)^(n - 1).
/// let law = KolmogorovSmirnovOneSided::new(5)?;
/// assert!((law.cdf(0.1_f64)? - 0.146_41).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.9)?)? - 0.9).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Z. W. Birnbaum and F. H. Tingey, "One-sided confidence contours for
/// probability distribution functions", *Ann. Math. Stat.* 22(4), 1951,
/// 592–596.
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovOneSided<S: RealScalar> {
    n: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> KolmogorovSmirnovOneSided<S> {
    /// Creates the one-sided law for sample size `n`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n = 0`.
    /// - `ScalarConversion` if `n` is not exactly representable as `f64`.
    pub fn new(n: usize) -> Result<Self, SciOxideError> {
        Self::with_options(n, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(n: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        count(n)?;
        Ok(Self { n, options })
    }
}
impl<S: RealScalar> KolmogorovSmirnovOneSided<S> {
    /// The one-sided kernel needs no matrices; `scratch` is unused.
    fn kernel_with(
        &self,
        x: f64,
        q: Query,
        _scratch: &mut KolmogorovWorkspace,
        w: &mut Work,
    ) -> Result<f64, SciOxideError> {
        smirnov(self.n, x, q, w)
    }
}
impl<S: RealScalar> Statistic<S> for KolmogorovSmirnovOneSided<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), Some(S::ONE))
    }
    fn median(&self) -> f64 {
        (core::f64::consts::LN_2 / (2. * self.n as f64)).sqrt()
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        smirnov(self.n, x, q, w)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.quantile(crate::float::open_word_probability(
            rng,
            self.options.sample_words,
        )?)
    }
}
statistic_law!(KolmogorovSmirnovOneSided, report);
fn smirnov(n: usize, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
    let nf = n as f64;
    if q == Query::Density && x == 0.0 {
        return Ok(0.0);
    }
    if q == Query::Density && x == 1.0 && n == 1 {
        return Ok(0.0);
    }
    if let Some(v) = boundary(x, 0.0, 1.0, q) {
        return Ok(v);
    }
    // Exact lower-end formula avoids subtracting a near-one Smirnov survival sum.
    if x <= 1.0 / nf {
        w.charge(1)?;
        let lc = x.ln() + (nf - 1.0) * x.ln_1p();
        return Ok(match q {
            Query::Lower => lc,
            Query::Upper => log1mexp(lc)?,
            Query::Density => (nf - 2.0) * x.ln_1p() + (nf * x).ln_1p(),
        });
    }
    // Birnbaum-Tingey exact finite sum for the one-sided statistic:
    // S(x) = x sum_j C(n, j) (x + j/n)^(j-1) (1 - x - j/n)^(n-j). Terms are
    // accumulated in log space with an incrementally updated ln C(n, j), and
    // the derivative (for the density) is carried as a weighted average
    // relative to the running total so it never overflows.
    let end = (nf * (1.0 - x)).floor() as usize;
    let mut total = f64::NEG_INFINITY;
    let mut derivative = 0.0;
    let mut log_binomial = 0.0;
    for j in 0..=end {
        w.charge(1)?;
        let jf = j as f64;
        let a = x + jf / nf;
        let b = 1.0 - a;
        let log = if j == 0 {
            nf * (-x).ln_1p()
        } else if b <= 0.0 {
            continue;
        } else {
            x.ln() + log_binomial + (jf - 1.0) * a.ln() + (nf - jf) * b.ln()
        };
        let next = logadd(total, log)?;
        if q == Query::Density {
            let rate = if j == 0 {
                -nf / (1.0 - x)
            } else {
                1.0 / x + (jf - 1.0) / a - (nf - jf) / b
            };
            derivative = derivative * (total - next).exp() + rate * (log - next).exp();
        }
        total = next;
        if j < end {
            log_binomial += (nf - jf).ln() - (jf + 1.0).ln();
        }
    }
    // The incremental log-binomial and log-sum roundoff grows with n and the
    // logarithmic magnitude; a fixed rounding floor understates extreme tails.
    w.error = w.error.max(16.0 * f64::EPSILON * (nf + total.abs()));
    Ok(match q {
        Query::Upper => total,
        Query::Lower => log1mexp(total)?,
        Query::Density => total + (-derivative).ln(),
    })
}

#[derive(Clone, Copy, Default, Debug)]
struct Dual {
    v: f64,
    d: f64,
}
// Monomorphized arithmetic: tail queries never construct or carry derivatives.
trait DurbinEntry: Copy + Default {
    fn new(value: f64, derivative: f64) -> Self;
    fn value(self) -> f64;
    fn derivative(self) -> f64;
    fn is_zero(self) -> bool {
        self.value() == 0.0 && self.derivative() == 0.0
    }
    fn add(&mut self, other: Self);
    fn add_product(&mut self, left: Self, right: Self);
    fn divide(&mut self, scale: f64);
}
impl DurbinEntry for f64 {
    fn new(value: f64, _: f64) -> Self {
        value
    }
    fn value(self) -> f64 {
        self
    }
    fn derivative(self) -> f64 {
        0.0
    }
    fn add(&mut self, other: Self) {
        *self += other;
    }
    fn add_product(&mut self, left: Self, right: Self) {
        *self += left * right;
    }
    fn divide(&mut self, scale: f64) {
        *self /= scale;
    }
}
impl DurbinEntry for Dual {
    fn new(v: f64, d: f64) -> Self {
        Self { v, d }
    }
    fn value(self) -> f64 {
        self.v
    }
    fn derivative(self) -> f64 {
        self.d
    }
    fn add(&mut self, other: Self) {
        self.v += other.v;
        self.d += other.d;
    }
    fn add_product(&mut self, left: Self, right: Self) {
        self.v += left.v * right.v;
        self.d += left.d * right.v + left.v * right.d;
    }
    fn divide(&mut self, scale: f64) {
        self.v /= scale;
        self.d /= scale;
    }
}

#[derive(Debug, Default)]
struct DurbinWorkspace<T> {
    matrix: Vec<T>,
    square: Vec<T>,
    row: Vec<T>,
    next_row: Vec<T>,
    dimension: usize,
}
impl<T: DurbinEntry> DurbinWorkspace<T> {
    const fn new() -> Self {
        Self {
            matrix: Vec::new(),
            square: Vec::new(),
            row: Vec::new(),
            next_row: Vec::new(),
            dimension: 0,
        }
    }
    fn buffers(&self, dimension: usize) -> Result<[(&Vec<T>, usize); 4], SciOxideError> {
        let len = matrix_len(dimension)?;
        Ok([
            (&self.matrix, len),
            (&self.square, len),
            (&self.row, dimension),
            (&self.next_row, dimension),
        ])
    }
    fn bytes(&self, dimension: usize) -> Result<usize, SciOxideError> {
        let elements = self
            .buffers(dimension)?
            .into_iter()
            .try_fold(0, |sum, (buffer, len)| {
                scioxide_core::backend::checked_add(
                    sum,
                    buffer.capacity().max(len),
                    scioxide_core::quantity!("ks_workspace"),
                )
            })?;
        crate::workspace::storage_bytes::<T>(elements)
    }
    fn reserve(&mut self, requirements: &KolmogorovRequirements) -> Result<(), SciOxideError> {
        crate::workspace::admit_bytes(self.bytes(requirements.dimension)?, requirements.max_bytes)?;
        let len = matrix_len(requirements.dimension)?;
        for (buffer, len) in [
            (&mut self.matrix, len),
            (&mut self.square, len),
            (&mut self.row, requirements.dimension),
            (&mut self.next_row, requirements.dimension),
        ] {
            if buffer.len() < len {
                scioxide_core::backend::try_reserve(
                    buffer,
                    len - buffer.len(),
                    scioxide_core::quantity!("ks_workspace"),
                )?;
                buffer.resize(len, T::default());
            }
        }
        self.dimension = self.dimension.max(requirements.dimension);
        crate::workspace::admit_bytes(self.bytes(0)?, requirements.max_bytes)
    }
}
/// Durbin matrix size needed by a [`KolmogorovSmirnovTwoSided`] tail query, with the byte limit of
/// the law.
///
/// See [`KolmogorovSmirnovTwoSided::workspace_requirements`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KolmogorovRequirements {
    dimension: usize,
    max_bytes: usize,
}
impl KolmogorovRequirements {
    /// Returns the Durbin matrix dimension, or zero where no matrix is needed.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the byte limit of the workspace, [`LawOptions::max_bytes`] of the law.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}
fn matrix_len(dimension: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(
        dimension,
        dimension,
        scioxide_core::quantity!("ks_workspace"),
    )
}
/// Reusable storage for [`KolmogorovSmirnovTwoSided::evaluate_tail`]: two `f64` matrices and two
/// rows.
///
/// Size it with [`KolmogorovSmirnovTwoSided::workspace_requirements`] and
/// [`Self::try_reserve`], or create it with
/// [`KolmogorovSmirnovTwoSided::workspace`]. See
/// [`KolmogorovSmirnovTwoSided::workspace_requirements`] for an example.
#[derive(Debug, Default)]
pub struct KolmogorovWorkspace {
    scratch: DurbinWorkspace<f64>,
}
impl KolmogorovWorkspace {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            scratch: DurbinWorkspace::new(),
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage already held exceeds the byte limit in
    ///   `requirements`.
    /// - `SizeOverflow` if the storage size overflows `usize`.
    pub fn check(&self, requirements: &KolmogorovRequirements) -> Result<Capacity, SciOxideError> {
        crate::workspace::admit_bytes(self.scratch.bytes(0)?, requirements.max_bytes)?;
        Ok(if self.scratch.dimension >= requirements.dimension {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the matrices and rows to `requirements`.
    ///
    /// The byte limit is checked before allocating. After a failure the
    /// workspace may keep capacity it has already grown.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage would exceed the byte limit.
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the allocation fails.
    pub fn try_reserve(
        &mut self,
        requirements: &KolmogorovRequirements,
    ) -> Result<(), SciOxideError> {
        self.scratch.reserve(requirements)
    }
}
/// Exact null distribution of the two-sided Kolmogorov–Smirnov statistic `Dₙ` for a sample of size
/// `n`.
///
/// `Dₙ = sup_x |Fₙ(x) − F(x)|` for the empirical distribution function `Fₙ`
/// of `n` independent draws from a continuous `F`. The support is
/// `[1/(2n), 1]`. For `x ≥ 1/2` the upper tail is twice the one-sided tail.
/// Where Miller's doubled one-sided approximation is within the tolerance, it
/// is used directly. Otherwise the distribution function is
/// `n!/nⁿ (Hⁿ)ₖₖ` for the `m × m` Durbin matrix `H`, with `k = ⌊n x⌋ + 1` and
/// `m = 2k − 1` (Marsaglia–Tsang–Wang). The matrix power is formed by repeated
/// squaring in O(m³ log n) time, and a tail needs `2m² + 2m` `f64` values of
/// storage.
///
/// The trait methods allocate this storage on each call. For repeated tail
/// queries, reuse a [`KolmogorovWorkspace`] with [`Self::evaluate_tail`].
/// Quantiles and draws repeat matrix evaluations and may need a work limit
/// above the [`LawOptions`] default.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, KolmogorovSmirnovOneSided, LawOptions};
/// use scioxide_random::{KolmogorovSmirnovTwoSided, Quantile, SeedableRng, SplitMix64, Survival};
/// let work = scioxide_core::Budget::ONE.saturating_add(1 << 24);
/// let options = LawOptions::default().with_work_budget(work);
/// let law = KolmogorovSmirnovTwoSided::with_options(20, options)?;
/// // At x >= 1/2 the one-sided exceedances are disjoint, doubling the one-sided tail.
/// let one = KolmogorovSmirnovOneSided::new(20)?;
/// assert!((law.sf(0.6_f64)? - 2.0 * one.sf(0.6)?).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.9)?)? - 0.9).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!((0.025..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. Marsaglia, W. W. Tsang and J. Wang, "Evaluating Kolmogorov's
///   distribution", *J. Stat. Softw.* 8(18), 2003.
/// - R. Simard and P. L'Ecuyer, "Computing the two-sided Kolmogorov–Smirnov
///   distribution", *J. Stat. Softw.* 39(11), 2011.
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovTwoSided<S: RealScalar> {
    n: usize,
    /// Smallest attainable statistic, `1/(2n)`.
    low: S,
    options: LawOptions<S>,
}
impl<S: RealScalar> KolmogorovSmirnovTwoSided<S> {
    /// Creates the two-sided law for sample size `n`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n = 0`.
    /// - `ScalarConversion` if `n` is not exactly representable as `f64`.
    pub fn new(n: usize) -> Result<Self, SciOxideError> {
        Self::with_options(n, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(n: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let low = number(0.5 / count(n)?)?;
        Ok(Self { n, low, options })
    }
    // Durbin dimension of a finite query; zero where boundary or series formulas apply.
    fn dimension(&self, x: f64) -> Result<usize, SciOxideError> {
        if x <= 1.0 / self.n as f64
            || x >= 0.5
            || miller_error(self.n as f64 * x * x)
                <= self
                    .options
                    .tolerance
                    .relative()
                    .into_f64()
                    .min(f64::SQRT_EPSILON)
        {
            return Ok(0);
        }
        let k = (self.n as f64 * x).floor() as usize + 1;
        k.checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("ks_workspace")))
    }
    fn requirements(&self, x: f64) -> Result<KolmogorovRequirements, SciOxideError> {
        Ok(KolmogorovRequirements {
            dimension: self.dimension(x)?,
            max_bytes: self.options.max_bytes(),
        })
    }
    /// Returns the workspace size that a tail query at `x` needs, with the byte limit of the law.
    ///
    /// ```
    /// use scioxide_random::{KolmogorovSmirnovTwoSided, LawOptions, LogCdf};
    /// let law = KolmogorovSmirnovTwoSided::new(20)?;
    /// let requirements = law.workspace_requirements(0.15)?;
    /// assert_eq!(requirements.dimension(), 7);
    /// let mut workspace = law.workspace(0.15)?;
    /// let tail = law.evaluate_tail(0.15, false, &mut workspace)?;
    /// assert_eq!(tail.log_probability, law.log_cdf(0.15)?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` is NaN or infinite.
    /// - `SizeOverflow` if the matrix dimension overflows `usize`.
    pub fn workspace_requirements(&self, x: S) -> Result<KolmogorovRequirements, SciOxideError> {
        argument(x, "x")?;
        self.requirements(x.into_f64())
    }
    /// Creates a workspace large enough for tail queries at `x`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `ResourceLimit` if the workspace would exceed [`LawOptions::max_bytes`].
    /// - `AllocationFailure` if the allocation fails.
    pub fn workspace(&self, x: S) -> Result<KolmogorovWorkspace, SciOxideError> {
        let mut workspace = KolmogorovWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(x)?)?;
        Ok(workspace)
    }
    /// Computes `ln P(Dₙ > x)` if `upper` is true, or `ln P(Dₙ ≤ x)` otherwise,
    /// in `workspace`, with an absolute error estimate and a work count.
    ///
    /// A tail above 15/16 is computed from the complementary tail. The error
    /// estimate includes the final rounding to `S`. The workspace is checked
    /// but never grown.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` is NaN or infinite.
    /// - `DimensionMismatch` if `workspace` is smaller than
    ///   [`Self::workspace_requirements`] at `x`.
    /// - `ResourceLimit` if `workspace` holds more than [`LawOptions::max_bytes`].
    /// - `ConvergenceFailure` if the evaluation exhausts the work limit.
    /// - `NumericalInstability` if the matrix power or a sum breaks down.
    pub fn evaluate_tail(
        &self,
        x: S,
        upper: bool,
        workspace: &mut KolmogorovWorkspace,
    ) -> Result<TestStatisticResult<S>, SciOxideError> {
        argument(x, "x")?;
        let x = x.into_f64();
        let requirements = self.requirements(x)?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(requirements.dimension),
                scioxide_core::Dimensions::one(workspace.scratch.dimension),
            ));
        }
        let mut work = Work::new(self.options);
        let (value, computed) =
            crate::law::smaller_tail_first(x >= self.median(), upper, |upper| {
                let query = if upper { Query::Upper } else { Query::Lower };
                two(self.n, x, query, &mut workspace.scratch, &mut work)
            })?;
        math::tail_result(value, computed.exp(), &work)
    }
}
impl<S: RealScalar> KolmogorovSmirnovTwoSided<S> {
    /// A query reusing `scratch`, which grows to its matrix dimension.
    fn kernel_with(
        &self,
        x: f64,
        q: Query,
        scratch: &mut KolmogorovWorkspace,
        w: &mut Work,
    ) -> Result<f64, SciOxideError> {
        if q == Query::Density {
            let mut scratch = DurbinWorkspace::<Dual>::new();
            scratch.reserve(&self.requirements(x)?)?;
            two(self.n, x, q, &mut scratch, w)
        } else {
            scratch.try_reserve(&self.requirements(x)?)?;
            two(self.n, x, q, &mut scratch.scratch, w)
        }
    }
}
impl<S: RealScalar> Statistic<S> for KolmogorovSmirnovTwoSided<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(self.low), Some(S::ONE))
    }
    fn median(&self) -> f64 {
        KOLMOGOROV_MEDIAN / (self.n as f64).sqrt()
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        self.kernel_with(x, q, &mut KolmogorovWorkspace::new(), w)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.quantile(crate::float::open_word_probability(
            rng,
            self.options.sample_words,
        )?)
    }
}
statistic_law!(KolmogorovSmirnovTwoSided);
fn square<T: DurbinEntry>(
    a: &[T],
    c: &mut [T],
    m: usize,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    // H is persymmetric: H^T = J H J for the reversal matrix J. Every
    // power retains this identity, so only one anti-triangle of its square
    // needs arithmetic; reflect the rest. The same identity holds for dH/dx.
    c.fill(T::default());
    for (i, (left, row)) in a.chunks_exact(m).zip(c.chunks_exact_mut(m)).enumerate() {
        let width = m - i;
        // The early powers retain exact Hessenberg zeros. Skip their updates,
        // without dropping small positive entries or derivative-only entries.
        let active = left.iter().filter(|x| !x.is_zero()).count();
        w.charge(active.checked_mul(width).ok_or_else(exhausted)?)?;
        for (&x, right) in left.iter().zip(a.chunks_exact(m)) {
            if x.is_zero() {
                continue;
            }
            for (out, &y) in row[..width].iter_mut().zip(right) {
                out.add_product(x, y);
            }
        }
    }
    for i in 0..m {
        for j in m - i..m {
            c[i * m + j] = c[(m - 1 - j) * m + m - 1 - i];
        }
    }
    normalize(c)
}
fn row_product<T: DurbinEntry>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    let active = a.iter().filter(|x| !x.is_zero()).count();
    w.charge(active.checked_mul(m).ok_or_else(exhausted)?)?;
    c.fill(T::default());
    for (&x, right) in a.iter().zip(b.chunks_exact(m)) {
        if x.is_zero() {
            continue;
        }
        for (out, &y) in c.iter_mut().zip(right) {
            out.add_product(x, y);
        }
    }
    normalize(c)
}
fn normalize<T: DurbinEntry>(values: &mut [T]) -> Result<f64, SciOxideError> {
    // Rescale every product to max-norm 1 and return the log of the scale;
    // raising the Durbin matrix to the n-th power would otherwise overflow.
    // The caller accumulates the logs, so the final entry is exact up to the
    // factored scale.
    let mut scale: f64 = 0.0;
    for x in values.iter() {
        scale = scale.max(finite(x.value())?.abs());
        finite(x.derivative())?;
    }
    crate::law::positive_result(scale)?;
    for x in values {
        x.divide(scale);
    }
    Ok(scale.ln())
}
// Conservative estimate for the omitted intersection term in Miller's
// doubled one-sided expansion (Simard–L'Ecuyer, upper-tail regime).
fn miller_error(t: f64) -> f64 {
    if t < 4.0 {
        1.0
    } else {
        16.0 * (1.0 + t).powi(2) * (-6.0 * t).exp()
    }
}
fn two<T: DurbinEntry>(
    n: usize,
    x: f64,
    q: Query,
    s: &mut DurbinWorkspace<T>,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    let nf = n as f64;
    let low = 0.5 / nf;
    if n == 1 && q == Query::Density && (0.5..=1.0).contains(&x) {
        return Ok(core::f64::consts::LN_2);
    }
    if let Some(v) = boundary(x, low, 1.0, q) {
        return Ok(v);
    }
    if x <= 1.0 / nf {
        w.charge(1)?;
        let t = 2.0 * x - 1.0 / nf;
        let lc = scioxide_special::ln_gamma(nf + 1.0)? + nf * t.ln();
        return Ok(match q {
            Query::Lower => lc,
            Query::Upper => log1mexp(lc)?,
            Query::Density => lc + (2.0 * nf / t).ln(),
        });
    }
    // Upper region: P(D_n >= x) = 2 P(D_n^+ >= x) minus an intersection term
    // that is exactly zero for x >= 1/2 and negligible once Miller's bound is
    // below the tolerance. This avoids the O(k^3 log n) matrix method there.
    let approximate = miller_error(nf * x * x);
    if x >= 0.5 || approximate <= w.tolerance {
        if x < 0.5 {
            w.error = w.error.max(approximate);
        }
        let v = smirnov(n, x, if q == Query::Density { q } else { Query::Upper }, w)?
            + core::f64::consts::LN_2;
        return Ok(if q == Query::Lower { log1mexp(v)? } else { v });
    }
    // Marsaglia-Tsang-Wang (2003) exact method: P(D_n < x) = n!/n^n (H^n)_{kk}
    // for the m x m Durbin matrix H with m = 2k - 1, k = floor(n x) + 1, h = k - n x in (0, 1].
    // Only the required row of H^n is accumulated. Density queries instantiate
    // this same kernel with dual numbers; tails use ordinary binary64 entries.
    let k = (nf * x).floor() as usize + 1;
    let m = k
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(exhausted)?;
    let size = matrix_len(m)?;
    if s.dimension < m {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(m),
            scioxide_core::Dimensions::one(s.dimension),
        ));
    }
    let (mut a, mut c, mut row, mut next_row) = (
        &mut s.matrix[..size],
        &mut s.square[..size],
        &mut s.row[..m],
        &mut s.next_row[..m],
    );
    a.fill(T::default());
    row.fill(T::default());
    let h = k as f64 - nf * x;
    // Entries are inverse factorials 1/(i - j + 1)!, formed along each row by
    // the running quotient recurrence instead of exp(-ln Gamma) per entry.
    for i in 0..m {
        w.charge(m)?;
        let mut inverse = 1.0;
        for j in (0..=(i + 1).min(m - 1)).rev() {
            let power = i + 1 - j;
            if power > 0 {
                inverse /= power as f64;
            }
            a[i * m + j] = T::new(inverse, 0.0);
        }
    }
    // Boundary corrections h^p / p! for p = 1..=m, with running 1/p!.
    let mut fact = 1.0;
    let mut h_power = 1.0;
    w.charge(m)?;
    for i in 0..m {
        let p = (i + 1) as f64;
        fact /= p;
        let derivative = nf * p * h_power * fact;
        h_power *= h;
        let correction = T::new(-h_power * fact, derivative);
        a[i * m].add(correction);
        a[(m - 1) * m + (m - 1 - i)].add(correction);
    }
    if h > 0.5 {
        let t = 2.0 * h - 1.0;
        // After the loop `fact` is 1/m!.
        a[(m - 1) * m].add(T::new(
            t.powf(m as f64) * fact,
            -2.0 * nf * m as f64 * t.powf(m as f64 - 1.0) * fact,
        ));
    }
    row[k - 1] = T::new(1.0, 0.0);
    // Binary exponentiation of H with separately tracked log scales for the
    // running power (`la`) and the accumulated row (`lb`).
    let mut e = n;
    let mut la = 0.0;
    let mut lb = 0.0;
    while e > 0 {
        if e % 2 == 1 {
            lb += la + row_product(row, a, next_row, m, w)?;
            core::mem::swap(&mut row, &mut next_row);
        }
        e /= 2;
        if e > 0 {
            la = 2.0 * la + square(a, c, m, w)?;
            core::mem::swap(&mut a, &mut c);
        }
    }
    let value = row[k - 1];
    let lc = value.value().ln() + lb + scioxide_special::ln_gamma(nf + 1.0)? - nf * nf.ln();
    w.error = w.error.max(128.0 * f64::EPSILON * nf);
    if q == Query::Density {
        return Ok(lc + (value.derivative() / value.value()).ln());
    }
    if q == Query::Lower {
        return crate::law::log_probability(lc);
    }
    // Complement only when its rounding uncertainty meets the requested tail accuracy.
    let upper = -lc.exp_m1();
    if upper <= 0.0 || w.error / upper > w.tolerance {
        return Err(crate::distribution::moment_math::cancellation());
    }
    w.error /= upper;
    Ok(upper.ln())
}
macro_rules! moments {
    ($law:ident) => {
        impl<S: RealScalar> $law<S> {
            fn moment(&self, order: f64, w: &mut Work) -> Result<f64, SciOxideError> {
                // E[X^r] = \int r x^(r-1) S(x) dx. The finite-n KS laws are
                // piecewise polynomial with kinks at multiples of 1/n, so the
                // integral is split at those knots to keep each panel smooth.
                // One workspace serves every quadrature node.
                let mut scratch = KolmogorovWorkspace::new();
                let mut sum = f64::NEG_INFINITY;
                for j in 0..self.n {
                    sum = logadd(
                        sum,
                        integral(
                            |x, w| {
                                Ok(order.ln()
                                    + (order - 1.0) * x.ln()
                                    + self.kernel_with(x, Query::Upper, &mut scratch, w)?)
                            },
                            j as f64 / self.n as f64,
                            (j + 1) as f64 / self.n as f64,
                            w,
                        )?,
                    )?;
                }
                Ok(sum.exp())
            }
        }
        impl<S: RealScalar> Mean<S> for $law<S> {
            fn mean(&self) -> Result<S, SciOxideError> {
                number(self.moment(1.0, &mut Work::new(self.options))?)
            }

            fn mean_scoped(
                &self,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| {
                    let mut work = Work::new(self.options);
                    work.limit = *left;
                    let w = &mut work;
                    let result = (|| number(self.moment(1.0, w)?))();
                    *left = work.limit - work.used;
                    result
                })
            }
        }
        impl<S: RealScalar> Variance<S> for $law<S> {
            fn variance(&self) -> Result<S, SciOxideError> {
                let mut w = Work::new(self.options);
                // Resolve the moments more tightly than their difference,
                // then reject cancellation using the accumulated error.
                w.tolerance = (w.tolerance / 128.0).max(64.0 * f64::EPSILON);
                let mean = self.moment(1.0, &mut w)?;
                let second = self.moment(2.0, &mut w)?;
                let value = crate::distribution::moment_math::variance_from_moments_with_accuracy(
                    second,
                    mean,
                    w.error,
                    self.options.tolerance.relative().into_f64(),
                )?;
                super::super::moment_math::variance(value.ln())
            }

            fn variance_scoped(
                &self,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| {
                    let mut work = Work::new(self.options);
                    work.limit = *left;
                    let w = &mut work;
                    let result = (|| {
                        // Resolve the moments more tightly than their difference,
                        // then reject cancellation using the accumulated error.
                        w.tolerance = (w.tolerance / 128.0).max(64.0 * f64::EPSILON);
                        let mean = self.moment(1.0, w)?;
                        let second = self.moment(2.0, w)?;
                        let value =
                            crate::distribution::moment_math::variance_from_moments_with_accuracy(
                                second,
                                mean,
                                w.error,
                                self.options.tolerance.relative().into_f64(),
                            )?;
                        super::super::moment_math::variance(value.ln())
                    })();
                    *left = work.limit - work.used;
                    result
                })
            }
        }
    };
}
moments!(KolmogorovSmirnovOneSided);
moments!(KolmogorovSmirnovTwoSided);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use scioxide_core::{NumericalIssue, SciOxideErrorKind};
    #[test]
    fn persymmetric_square_preserves_values_and_derivatives() {
        // Anti-diagonal reflection symmetry, without ordinary symmetry.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 6.0, 4.0, 1.0].map(|v| Dual::new(v, 0.1 * v));
        let expected = [27.0, 24.0, 10.0, 36.0, 41.0, 24.0, 28.0, 36.0, 27.0];
        let mut output = [Dual::default(); 9];
        let mut work = Work::new(LawOptions::<f64>::default());
        let scale = square(&a, &mut output, 3, &mut work).unwrap();
        assert!((scale - 41.0_f64.ln()).abs() < 1e-15);
        assert_eq!(work.used, 18);
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual.v - expected / 41.0).abs() < 1e-15);
            assert!((actual.d - 0.2 * expected / 41.0).abs() < 1e-15);
        }
    }
    #[test]
    fn durbin_workspace_size_overflow_precedes_allocation() {
        let mut scratch = KolmogorovWorkspace::new();
        assert_eq!(
            scratch
                .try_reserve(&KolmogorovRequirements {
                    dimension: usize::MAX,
                    max_bytes: usize::MAX,
                })
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(scratch.scratch.bytes(0).unwrap(), 0);
    }
    #[test]
    fn durbin_product_range_failure_is_not_budget_exhaustion() {
        for (value, issue) in [
            (0.0, NumericalIssue::Underflow),
            (f64::MAX, NumericalIssue::Overflow),
        ] {
            let a = [Dual { v: value, d: 0.0 }];
            let mut out = [Dual { v: 0.0, d: 0.0 }];
            crate::error_assertions::numerical(
                square(
                    &a,
                    &mut out,
                    1,
                    &mut Work::new(LawOptions::<f64>::default()),
                )
                .unwrap_err(),
                issue,
            );
        }
        let a = [Dual { v: 1.0, d: 0.0 }; 4];
        let mut out = a;
        let mut work = Work::new(LawOptions::<f64>::default());
        work.limit = 1;
        assert_eq!(
            square(&a, &mut out, 2, &mut work).unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
    }
    #[test]
    fn unresolved_durbin_complement_is_cancellation() {
        let law = KolmogorovSmirnovTwoSided::<f64>::new(20).unwrap();
        let mut scratch = law.workspace(0.2).unwrap();
        let mut work = Work::new(LawOptions::<f64>::default());
        work.error = 1.0; // A propagated error larger than the complemented tail.
        crate::error_assertions::numerical(
            two(20, 0.2, Query::Upper, &mut scratch.scratch, &mut work).unwrap_err(),
            NumericalIssue::Cancellation,
        );
    }
}
