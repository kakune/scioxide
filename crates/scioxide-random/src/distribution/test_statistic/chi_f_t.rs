//! F, noncentral chi-squared, noncentral F and noncentral t distributions.
use super::*;
use crate::distribution::LawOptions;
use crate::distribution::chi_squared::ChiSquareKernel;
use crate::distribution::normal::standard_normal;
use crate::distribution::poisson::PoissonSampler;
use math::{beta_log, boundary, normal_log, real_integral};

// Specialized inverses keep their working coordinate in binary64, as the tail
// kernels do. Rice reuses the chi-square inverse before taking its square root.
pub(in crate::distribution) fn inverse_result<S: RealScalar>(
    x: f64,
    p: f64,
    positive: bool,
) -> Result<S, SciOxideError> {
    if (p == 0.0 || p == 1.0) && x == f64::INFINITY {
        return Ok(S::INFINITY);
    }
    if p > 0.0 && p < 1.0 && x.abs() > S::MAX.into_f64() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "interior quantile exceeds the scalar range",
        ));
    }
    let value = number::<S>(x)?;
    if p > 0.0 && p < 1.0 && (!value.is_finite() || (positive && value <= S::ZERO)) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "interior quantile cannot be resolved in the scalar range",
        ));
    }
    Ok(value)
}

fn inverse_endpoint(p: f64, upper: bool, positive: bool) -> Result<Option<f64>, SciOxideError> {
    crate::law::probability(p, scioxide_core::argument!("p"))?;
    Ok(if p == 0.0 || p == 1.0 {
        Some(if (p == 0.0) != upper {
            if positive { 0.0 } else { f64::NEG_INFINITY }
        } else {
            f64::INFINITY
        })
    } else {
        None
    })
}

// F = (b/a) z/(1-z), with z ~ Beta(a/2,b/2). Reflect both the
// probability and the beta coordinate so a small complement is never lost.
fn f_inverse(a: f64, b: f64, p: f64, upper: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    if let Some(x) = inverse_endpoint(p, upper, true)? {
        return Ok(x);
    }
    let (log_p, upper) = if p > 0.5 {
        ((-p).ln_1p(), !upper)
    } else {
        (p.ln(), upper)
    };
    let (aa, bb) = if upper { (b, a) } else { (a, b) };
    let z = w.special(|budget| {
        scioxide_special::beta_inc_inv_log_with_options(
            aa / 2.0,
            bb / 2.0,
            log_p,
            scioxide_special::BetaOptions::new(budget),
        )
    })?;
    let log_ratio = if z <= 0.5 {
        z.ln() - (-z).ln_1p()
    } else {
        let y = w.special(|budget| {
            scioxide_special::beta_inc_inv_log_with_options(
                bb / 2.0,
                aa / 2.0,
                log1mexp(log_p)?,
                scioxide_special::BetaOptions::new(budget),
            )
        })?;
        (-y).ln_1p() - y.ln()
    };
    finite((b.ln() - a.ln() + if upper { -log_ratio } else { log_ratio }).exp())
}

// Safeguarded Newton inversion in `ln x` (positive laws) or `asinh x`
// coordinates, from a law-specific initial approximation, within the caller's
// remaining work.
fn smooth_inverse<S: RealScalar, T: Statistic<S>>(
    law: &T,
    positive: bool,
    p: f64,
    upper: bool,
    left: &mut usize,
    initial: impl FnOnce(f64, bool, &mut Work) -> Result<f64, SciOxideError>,
) -> Result<f64, SciOxideError> {
    if let Some(x) = inverse_endpoint(p, upper, positive)? {
        return Ok(x);
    }
    let (log_p, upper) = if p > 0.5 {
        ((-p).ln_1p(), !upper)
    } else {
        (p.ln(), upper)
    };
    let options = law.options();
    // An approximation can overflow even when the law's inverse is finite.
    // Its failure only discards the proposed start; actual tail and density
    // failures still propagate.
    let mut initial_work = Work::new(options);
    initial_work.limit = *left;
    let start = match initial(log_p, upper, &mut initial_work) {
        Ok(start) => start,
        Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::ConvergenceFailure => {
            return Err(error);
        }
        Err(_) => law.median(),
    };
    let mut remaining = initial_work.limit - initial_work.used;
    let guess = if positive {
        start.ln()
    } else if start.abs() > f64::MAX.sqrt() {
        (start.abs().ln() + core::f64::consts::LN_2).copysign(start)
    } else {
        start.asinh()
    };
    let guess = if guess.is_finite() { guess } else { 0.0 };
    let max_log = f64::MAX.ln();
    let coordinate = |y: f64| {
        if positive {
            y.exp()
        } else if y.abs() > max_log {
            // sinh(y) remains finite beyond the range of exp(y).
            // The omitted exp(-|y|)/2 is below a coordinate ULP.
            (y.abs() - core::f64::consts::LN_2).exp().copysign(y)
        } else {
            y.sinh()
        }
    };
    let evaluate = |y: f64, query, left: &mut usize| {
        Work::within(options, left, |work| {
            let x = finite(coordinate(y))?;
            if query == Query::Density {
                law.kernel(x, query, work)
            } else {
                law.tail_parts(x, query == Query::Upper, work).map(|v| v.0)
            }
        })
    };
    let inversion = crate::inversion::InversionOptions::default()
        .with_root_options(
            scioxide_rootfind::ScalarRootOptions::default()
                .with_max_iterations(options.work)
                .with_step_tolerance(scioxide_core::StepTolerance::new(
                    f64::EPSILON,
                    f64::EPSILON,
                )?)
                .with_residual_tolerance(scioxide_core::ResidualTolerance::new(
                    (options
                        .tolerance
                        .relative()
                        .into_f64()
                        .min(f64::SQRT_EPSILON)
                        * 0.125)
                        .max(8.0 * f64::EPSILON * (1.0 + log_p.abs())),
                    0.0,
                )?),
        )
        .with_work_budget(options.work);
    let bound = if positive {
        max_log
    } else {
        (max_log + core::f64::consts::LN_2).next_down()
    };
    let y = crate::inversion::invert_smooth_remaining(
        log_p,
        upper,
        guess,
        (
            if positive {
                f64::from_bits(1).ln()
            } else {
                -bound
            },
            bound,
        ),
        &inversion,
        |y, upper, left| evaluate(y, if upper { Query::Upper } else { Query::Lower }, left),
        |y, left| {
            Ok(evaluate(y, Query::Density, left)?
                + if positive {
                    y
                } else {
                    coordinate(y).hypot(1.0).ln()
                })
        },
        &mut remaining,
    )?;
    *left = remaining;
    finite(coordinate(y))
}

/// Fisher–Snedecor F distribution with `d₁ > 0` numerator and `d₂ > 0` denominator degrees of
/// freedom.
///
/// `X = (U/d₁) / (V/d₂)` for independent chi-squared variates `U` and `V`
/// with `d₁` and `d₂` degrees of freedom. The distribution function is
/// `F(x) = I_y(d₁/2, d₂/2)` with `y = d₁x / (d₁x + d₂)`, where `I` is the
/// regularized incomplete beta function. Quantiles invert the smaller of the
/// two tails, so they keep their accuracy near both ends. Draws form the ratio
/// of two chi-squared variates in log space, numerator first. A draw that
/// rounds to zero or overflows is a `NumericalInstability` error.
///
/// ```
/// use scioxide_random::{Cdf, Density, Distribution, FisherF, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// // Two and two degrees: F(x) = x / (1 + x).
/// let law = FisherF::new(2.0_f64, 2.0)?;
/// assert!((law.pdf(1.0)? - 0.25).abs() < 1e-14);
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-14 && (law.sf(3.0)? - 0.25).abs() < 1e-14);
/// assert!((law.quantile(0.75)? - 3.0).abs() < 1e-12);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct FisherF<S: RealScalar> {
    a: f64,
    b: f64,
    ratio: f64,
    log_ratio: f64,
    log_beta: Result<f64, SciOxideError>,
    numerator_chi: ChiSquareKernel<f64>,
    denominator_chi: ChiSquareKernel<f64>,
    options: LawOptions<S>,
}
impl<S: RealScalar> FisherF<S> {
    /// Creates an F law with `numerator` and `denominator` degrees of freedom.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a degree is NaN or infinite.
    /// - `DomainError` if a degree is not positive.
    pub fn new(numerator: S, denominator: S) -> Result<Self, SciOxideError> {
        Self::with_options(numerator, denominator, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        numerator: S,
        denominator: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let a = positive(numerator, scioxide_core::argument!("numerator"))?.into_f64();
        let b = positive(denominator, scioxide_core::argument!("denominator"))?.into_f64();
        Ok(Self {
            a,
            b,
            ratio: a / b,
            log_ratio: a.ln() - b.ln(),
            log_beta: scioxide_special::ln_beta(a / 2.0, b / 2.0),
            numerator_chi: ChiSquareKernel::new(a)?,
            denominator_chi: ChiSquareKernel::new(b)?,
            options,
        })
    }
}
impl<S: RealScalar> Statistic<S> for FisherF<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn median(&self) -> f64 {
        math::chi_square_center(self.a, self.a)
            / self.a
            / (math::chi_square_center(self.b, self.b) / self.b)
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        w.charge(1)?;
        if x == 0.0 && q == Query::Density {
            return Ok(if self.a < 2.0 {
                f64::INFINITY
            } else if self.a == 2.0 {
                self.log_ratio - self.log_beta.clone()?
            } else {
                f64::NEG_INFINITY
            });
        }
        if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
            return Ok(v);
        }
        let (lz, ly) = f_log_coordinates(x.ln() + self.log_ratio)?;
        match q {
            Query::Lower => beta_log(self.a / 2.0, self.b / 2.0, lz, ly, w),
            Query::Upper => beta_log(self.b / 2.0, self.a / 2.0, ly, lz, w),
            Query::Density => {
                Ok(self.a / 2.0 * lz + self.b / 2.0 * ly - x.ln() - self.log_beta.clone()?)
            }
        }
    }
    fn ordinary(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(if upper { S::ONE } else { S::ZERO });
        }
        let scaled = x.into_f64() * self.ratio;
        if !scaled.is_finite() || scaled < f64::MIN_POSITIVE || self.ratio < f64::MIN_POSITIVE {
            return Ok(if upper {
                self.log_sf(x)?
            } else {
                self.log_cdf(x)?
            }
            .exp());
        }
        let reflected = scaled > 1.0;
        let (a, b, coordinate) = if reflected {
            (self.b / 2.0, self.a / 2.0, 1.0 / (1.0 + scaled))
        } else {
            (self.a / 2.0, self.b / 2.0, scaled / (1.0 + scaled))
        };
        if coordinate < f64::MIN_POSITIVE {
            return Ok(if upper {
                self.log_sf(x)?
            } else {
                self.log_cdf(x)?
            }
            .exp());
        }
        let mut work = Work::new(self.options);
        work.charge(1)?;
        let value = work.special(|budget| {
            scioxide_special::beta_inc_with_options(
                a,
                b,
                coordinate,
                scioxide_special::BetaOptions::new(budget),
            )
        })?;
        if upper == reflected {
            number(value)
        } else if value <= 0.5 {
            number(1.0 - value)
        } else {
            // A near-one beta probability cannot safely be complemented.
            // Reuse the same ledger for the logarithmic tail evaluation.
            number(self.tail_parts(x.into_f64(), upper, &mut work)?.0.exp())
        }
    }
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        if let Some(x) = inverse_endpoint(p, upper, true)? {
            return inverse_result(x, p, true);
        }
        let x = Work::within(self.options, left, |work| {
            work.charge(1)?;
            f_inverse(self.a, self.b, p, upper, work)
        })?;
        inverse_result(x, p, true)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        crate::law::positive_draw(
            self.numerator_chi
                .sample_log_ratio(&self.denominator_chi, &mut r)?
                + self.b.ln()
                - self.a.ln(),
        )
    }
}
statistic_law!(FisherF, report);
fn f_log_coordinates(r: f64) -> Result<(f64, f64), SciOxideError> {
    Ok((-logadd(0.0, -r)?, -logadd(0.0, r)?))
}
fn f_value(a: f64, b: f64, j: f64, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
    if x == 0.0 && q == Query::Density {
        return Ok(if a + 2.0 * j < 2.0 {
            f64::INFINITY
        } else if a + 2.0 * j == 2.0 {
            (a / b).ln() - scioxide_special::ln_beta(a / 2.0 + j, b / 2.0)?
        } else {
            f64::NEG_INFINITY
        });
    }
    if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
        return Ok(v);
    }
    // F(x) = I_z(a/2 + j, b/2) with z = a x / (a x + b). Both ln z and
    // ln(1 - z) come from softplus of r = ln(a x / b), so neither side is
    // formed as 1 - (near one), and the upper tail uses the reflected beta.
    // `j` shifts the numerator shape for the Poisson terms of the noncentral F.
    let r = x.ln() + a.ln() - b.ln();
    let (lz, ly) = f_log_coordinates(r)?;
    match q {
        Query::Lower => beta_log(a / 2.0 + j, b / 2.0, lz, ly, w),
        Query::Upper => beta_log(b / 2.0, a / 2.0 + j, ly, lz, w),
        Query::Density => Ok((a / 2.0 + j) * lz + b / 2.0 * ly
            - x.ln()
            - scioxide_special::ln_beta(a / 2.0 + j, b / 2.0)?),
    }
}
impl<S: RealScalar> Mean<S> for FisherF<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.b <= 2.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        number(self.b / (self.b - 2.0))
    }
}
impl<S: RealScalar> Variance<S> for FisherF<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.b <= 4.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        number(
            2.0 * (self.b / (self.b - 2.0)).powi(2) * (1.0 + (self.b - 2.0) / self.a)
                / (self.b - 4.0),
        )
    }
}

/// Prepared noncentral chi-square variate with fixed degrees and noncentrality.
#[derive(Debug, Clone)]
pub(in crate::distribution) enum NoncentralChi {
    Central(ChiSquareKernel<f64>),
    /// `nu > 1`: `(Z + sqrt(lambda))^2 + chi^2_{nu - 1}`, the cheapest exact
    /// construction.
    Shifted {
        root: f64,
        rest: ChiSquareKernel<f64>,
    },
    /// `nu <= 1`, where the central part would need a nonpositive degree:
    /// `chi^2_{nu + 2J}` with `J ~ Poisson(lambda / 2)`.
    Mixture {
        nu: f64,
        count: PoissonSampler,
    },
}
impl NoncentralChi {
    pub(in crate::distribution) fn new(nu: f64, lambda: f64) -> Result<Self, SciOxideError> {
        Ok(if lambda == 0.0 {
            Self::Central(ChiSquareKernel::new(nu)?)
        } else if nu > 1.0 {
            Self::Shifted {
                root: lambda.sqrt(),
                rest: ChiSquareKernel::new(nu - 1.0)?,
            }
        } else {
            Self::Mixture {
                nu,
                count: crate::Poisson::new(lambda / 2.0)?.prepare_sampler()?,
            }
        })
    }
    fn sample<R: Rng<f64>>(&self, r: &mut R) -> Result<f64, SciOxideError> {
        match self {
            Self::Central(chi) => chi.sample(r),
            Self::Shifted { root, rest } => {
                let z = standard_normal::<f64, R>(r)? + root;
                finite(z * z + rest.sample(r)?)
            }
            Self::Mixture { nu, count } => {
                let j = crate::CountDistribution::sample(count, r)?;
                ChiSquareKernel::new(nu + 2.0 * j as f64)?.sample(r)
            }
        }
    }
}

/// Sum a Poisson mixture around the maximum *weighted integrand*, rather than
/// assuming the Poisson mode is also the mode in an extreme probability tail.
pub(in crate::distribution) fn mixture(
    mu: f64,
    mut f: impl FnMut(f64, &mut Work) -> Result<f64, SciOxideError>,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    if mu == 0.0 {
        w.charge(1)?;
        return f(0.0, w);
    }
    if !mu.is_finite() || mu > 4_503_599_627_370_496.0 {
        return Err(exhausted());
    }
    // Poisson(mu) mixture sum_j P(J = j) f(j) in log space. Terms are
    // log-concave in j, so the sum is started at the peak term and extended in
    // both directions; each side stops once the geometric bound on the rest,
    // v / (1 - ratio), falls below the tolerance relative to the total.
    let mut term = |j: usize, w: &mut Work| {
        w.charge(1)?;
        let k = j as f64;
        Ok::<_, SciOxideError>(super::super::discrete_math::poisson_log_mass(k, mu) + f(k, w)?)
    };
    let mut mode = super::super::count_math::count(mu.floor())?;
    let right_index = mode.checked_add(1).ok_or_else(exhausted)?;
    let mut peak = term(mode, w)?;
    // Locate the log-concave mixture peak by bracket expansion and integer search.
    let left = if mode > 0 {
        term(mode - 1, w)?
    } else {
        f64::NEG_INFINITY
    };
    let right = term(right_index, w)?;
    let (mut lo, mut hi) = if right > peak {
        let mut lo = mode;
        let mut hi = right_index;
        let mut step = 1usize;
        let mut old = right;
        loop {
            step = step.checked_mul(2).ok_or_else(exhausted)?;
            let next = hi.checked_add(step).ok_or_else(exhausted)?;
            let v = term(next, w)?;
            if v <= old {
                break (lo, next);
            }
            lo = hi;
            hi = next;
            old = v;
        }
    } else if left > peak {
        (0, mode)
    } else {
        (mode, mode)
    };
    while hi.saturating_sub(lo) > 2 {
        let m = lo + (hi - lo) / 2;
        if term(m, w)? < term(m + 1, w)? {
            lo = m;
        } else {
            hi = m + 1;
        }
    }
    for j in lo..=hi {
        let v = term(j, w)?;
        if v > peak {
            peak = v;
            mode = j;
        }
    }
    let mut total = peak;
    for direction in [-1isize, 1] {
        let mut j = mode;
        let mut previous = peak;
        loop {
            if direction < 0 && j == 0 {
                break;
            }
            j = if direction < 0 {
                j - 1
            } else {
                j.checked_add(1).ok_or_else(exhausted)?
            };
            let v = term(j, w)?;
            total = logadd(total, v)?;
            let ratio = v - previous;
            if v == f64::NEG_INFINITY
                || (ratio < 0.0 && v - log1mexp(ratio)? < total + (w.tolerance * 0.125).ln())
            {
                break;
            }
            previous = v;
        }
    }
    w.error = w.error.max(w.tolerance);
    Ok(total)
}

/// Noncentral chi-squared distribution with `ν > 0` degrees of freedom and noncentrality `λ ≥ 0`.
///
/// For integer `ν`, `X = Σ (Zᵢ + μᵢ)²` with independent standard normal `Zᵢ`
/// and `λ = Σ μᵢ²`. In general `X` is a Poisson mixture of central
/// chi-squared variates with `ν + 2J` degrees of freedom, `J ~ Poisson(λ/2)`.
/// The mean is `ν + λ` and the variance `2(ν + 2λ)`.
///
/// Tails sum the Poisson mixture of incomplete gamma tails with the
/// Benton–Krishnamoorthy recurrences, adding only positive terms. Quantiles
/// start from Pearson's approximation and refine the smaller log tail with a
/// safeguarded Newton iteration. `λ = 0` uses the central law. For `ν > 1`,
/// draws are `(Z + √λ)²` plus a central chi-squared variate with `ν − 1`
/// degrees of freedom. For `ν ≤ 1` they use the Poisson mixture, and a `λ/2`
/// beyond the range of [`Poisson::prepare_sampler`](crate::Poisson::prepare_sampler)
/// makes sampling, and only sampling, fail with `NotImplemented`.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Mean, NoncentralChiSquared, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival, Variance};
/// let law = NoncentralChiSquared::new(4.0_f64, 2.0)?;
/// assert!((law.mean()? - 6.0).abs() < 1e-14 && (law.variance()? - 16.0).abs() < 1e-14);
/// assert!((law.cdf(5.0)? + law.sf(5.0)? - 1.0).abs() < 1e-12);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. Benton and K. Krishnamoorthy, "Computing discrete mixtures of continuous
/// distributions: noncentral chisquare, noncentral t and the distribution of the
/// square of the sample multiple correlation coefficient", *Comput. Stat. Data
/// Anal.* 43(2), 2003, 249–267.
#[derive(Debug, Clone)]
pub struct NoncentralChiSquared<S: RealScalar> {
    pub(in crate::distribution) nu: f64,
    pub(in crate::distribution) lambda: f64,
    /// Prepared variate; an unsupported Poisson rate fails only sampling.
    pub(in crate::distribution) draw: Result<NoncentralChi, SciOxideError>,
    /// The central law of the same degrees, for zero noncentrality and the
    /// density at the origin.
    pub(in crate::distribution) central: Result<crate::ChiSquared<f64>, SciOxideError>,
    pub(in crate::distribution) options: LawOptions<S>,
}
impl<S: RealScalar> NoncentralChiSquared<S> {
    /// Creates a noncentral chi-squared law with `degrees` degrees of freedom and noncentrality
    /// `noncentrality`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if `degrees ≤ 0` or `noncentrality < 0`.
    pub fn new(degrees: S, noncentrality: S) -> Result<Self, SciOxideError> {
        Self::with_options(degrees, noncentrality, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        degrees: S,
        noncentrality: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(noncentrality, "noncentrality")?;
        if noncentrality < S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("noncentrality"),
                "noncentrality must be nonnegative",
            ));
        }
        let nu = positive(degrees, scioxide_core::argument!("degrees"))?.into_f64();
        let lambda = noncentrality.into_f64();
        Ok(Self {
            nu,
            lambda,
            draw: NoncentralChi::new(nu, lambda),
            central: options
                .widened()
                .and_then(|widened| crate::ChiSquared::with_options(nu, widened)),
            options,
        })
    }
}
impl<S: RealScalar> Statistic<S> for NoncentralChiSquared<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn median(&self) -> f64 {
        math::chi_square_center(self.nu + self.lambda, self.nu + 2. * self.lambda)
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if x == 0.0 && q == Query::Density {
            return Ok(self.central.as_ref().map_err(Clone::clone)?.log_pdf(x)? - self.lambda / 2.0);
        }
        if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
            return Ok(v);
        }
        // Poisson(lambda/2) mixture of chi-square laws with nu + 2j degrees:
        // incomplete gamma tails P(nu/2 + j, x/2) and densities d(nu/2 + j - 1) / 2.
        let family = super::mixture::Family::Gamma { y: 0.5 * x };
        let shape = 0.5 * self.nu;
        if self.lambda == 0.0 {
            w.charge(1)?;
            return match q {
                Query::Density => Ok(super::mixture::central_log_term(family, shape - 1.0)?
                    - core::f64::consts::LN_2),
                Query::Lower | Query::Upper => {
                    super::mixture::central_log_tail(family, shape, q == Query::Upper, w)
                }
            };
        }
        let weights = super::mixture::Weights {
            mu: 0.5 * self.lambda,
            offset: 0.0,
        };
        match q {
            Query::Density => {
                Ok(super::mixture::log_density(weights, shape, family, w)?
                    - core::f64::consts::LN_2)
            }
            Query::Lower | Query::Upper => {
                match super::mixture::log_tail(weights, shape, family, q == Query::Upper, w)? {
                    Some(value) => Ok(value),
                    // Beyond the recurrence range, sum the terms one by one.
                    None => mixture(
                        self.lambda / 2.0,
                        |j, w| {
                            w.charge(1)?;
                            super::mixture::central_log_tail(
                                super::mixture::Family::Gamma { y: 0.5 * x },
                                shape + j,
                                q == Query::Upper,
                                w,
                            )
                        },
                        w,
                    ),
                }
            }
        }
    }
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        inverse_result(self.inverse_value(p, upper, left)?, p, true)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        number(self.draw.as_ref().map_err(Clone::clone)?.sample(&mut r)?)
    }
}
statistic_law!(NoncentralChiSquared, report);
impl<S: RealScalar> NoncentralChiSquared<S> {
    pub(in crate::distribution) fn inverse_value(
        &self,
        p: f64,
        upper: bool,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        if let Some(x) = inverse_endpoint(p, upper, true)? {
            return Ok(x);
        }
        if self.lambda == 0.0 {
            crate::law::charge(left)?;
            let law = self.central.as_ref().map_err(Clone::clone)?;
            return if upper { law.isf(p) } else { law.quantile(p) };
        }
        smooth_inverse(self, true, p, upper, left, |log_p, upper, w| {
            self.initial(log_p, upper, w)
        })
    }
    fn initial(&self, log_p: f64, upper: bool, w: &mut Work) -> Result<f64, SciOxideError> {
        // Pearson (1959), Biometrika 46:364: match the first three cumulants
        // with a shifted, scaled central chi-square. Near zero use the leading
        // lower-tail term exp(-lambda/2) (x/2)^(nu/2) / Gamma(nu/2+1).
        let c = 1.0 + self.lambda / (self.nu + 2.0 * self.lambda);
        let degrees = (self.nu + 2.0 * self.lambda) / (c * c);
        let shape = degrees / 2.0;
        let central = if upper {
            w.special(|budget| {
                scioxide_special::gamma_q_inv_log_with_options(
                    shape,
                    log_p,
                    scioxide_special::GammaOptions::new(budget),
                )
            })?
        } else {
            w.special(|budget| {
                scioxide_special::gamma_p_inv_log_with_options(
                    shape,
                    log_p,
                    scioxide_special::GammaOptions::new(budget),
                )
            })?
        };
        let x = 2.0 * c * central - self.lambda * (self.lambda / (self.nu + 3.0 * self.lambda));
        if x > 0.0 && x.is_finite() {
            return Ok(x);
        }
        let lower = if upper { log1mexp(log_p)? } else { log_p };
        Ok((core::f64::consts::LN_2
            + (lower + self.lambda / 2.0 + scioxide_special::ln_gamma(self.nu / 2.0 + 1.0)?)
                / (self.nu / 2.0))
            .exp())
    }
}
impl<S: RealScalar> Mean<S> for NoncentralChiSquared<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(finite(self.nu + self.lambda)?)
    }
}
impl<S: RealScalar> Variance<S> for NoncentralChiSquared<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        number(finite(2.0 * (self.nu + 2.0 * self.lambda))?)
    }
}

/// Noncentral F distribution with `d₁ > 0` and `d₂ > 0` degrees of freedom and noncentrality `λ ≥
/// 0`.
///
/// `X = (U/d₁) / (V/d₂)`, where `U` is noncentral chi-squared with `d₁`
/// degrees of freedom and noncentrality `λ`, and `V` is an independent central
/// chi-squared variate with `d₂` degrees of freedom (singly noncentral F).
/// Tails sum the Poisson mixture of incomplete beta tails with the
/// Benton–Krishnamoorthy recurrences. Quantiles start from a scaled central F
/// with matched moments and refine the smaller log tail with a safeguarded
/// Newton iteration. `λ = 0` uses the central law.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, InverseSurvival, Mean, NoncentralF};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// let law = NoncentralF::new(4.0_f64, 10.0, 2.0)?;
/// assert!((law.mean()? - 1.875).abs() < 1e-14); // d2 (d1 + lambda) / (d1 (d2 - 2))
/// assert!((law.cdf(1.0)? + law.sf(1.0)? - 1.0).abs() < 1e-12);
/// assert!((law.sf(law.isf(0.05)?)? - 0.05).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. Benton and K. Krishnamoorthy, "Computing discrete mixtures of continuous
/// distributions: noncentral chisquare, noncentral t and the distribution of the
/// square of the sample multiple correlation coefficient", *Comput. Stat. Data
/// Anal.* 43(2), 2003, 249–267.
#[derive(Debug, Clone)]
pub struct NoncentralF<S: RealScalar> {
    a: f64,
    b: f64,
    lambda: f64,
    numerator_chi: Result<NoncentralChi, SciOxideError>,
    denominator_chi: ChiSquareKernel<f64>,
    options: LawOptions<S>,
}
impl<S: RealScalar> NoncentralF<S> {
    /// Creates a noncentral F law with `numerator` and `denominator` degrees of freedom and
    /// noncentrality `noncentrality`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if a degree is not positive or `noncentrality < 0`.
    pub fn new(numerator: S, denominator: S, noncentrality: S) -> Result<Self, SciOxideError> {
        Self::with_options(numerator, denominator, noncentrality, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        numerator: S,
        denominator: S,
        noncentrality: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(noncentrality, "noncentrality")?;
        if noncentrality < S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("noncentrality"),
                "noncentrality must be nonnegative",
            ));
        }
        let a = positive(numerator, scioxide_core::argument!("numerator"))?.into_f64();
        let b = positive(denominator, scioxide_core::argument!("denominator"))?.into_f64();
        let lambda = noncentrality.into_f64();
        Ok(Self {
            a,
            b,
            lambda,
            numerator_chi: NoncentralChi::new(a, lambda),
            denominator_chi: ChiSquareKernel::new(b)?,
            options,
        })
    }
}
impl<S: RealScalar> Statistic<S> for NoncentralF<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn median(&self) -> f64 {
        math::chi_square_center(self.a + self.lambda, self.a + 2. * self.lambda)
            / self.a
            / (math::chi_square_center(self.b, self.b) / self.b)
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if x <= 0.0 {
            return Ok(f_value(self.a, self.b, 0.0, x, q, w)?
                - if q == Query::Density {
                    self.lambda / 2.0
                } else {
                    0.0
                });
        }
        if self.lambda == 0.0 {
            return f_value(self.a, self.b, 0.0, x, q, w);
        }
        // Poisson(lambda/2) mixture of beta tails I_z(a/2 + j, b/2), z = ax/(ax+b);
        // the component densities are (a/2 + j) d(a/2 + j) / x.
        let (lz, ly) = f_log_coordinates(x.ln() + self.a.ln() - self.b.ln())?;
        let family = super::mixture::Family::Beta {
            b: 0.5 * self.b,
            lz,
            ly,
        };
        let weights = super::mixture::Weights {
            mu: 0.5 * self.lambda,
            offset: 0.0,
        };
        match q {
            Query::Density => {
                Ok(super::mixture::log_density(weights, 0.5 * self.a, family, w)? - x.ln())
            }
            Query::Lower | Query::Upper => {
                match super::mixture::log_tail(weights, 0.5 * self.a, family, q == Query::Upper, w)?
                {
                    Some(value) => Ok(value),
                    // Beyond the recurrence range, sum the terms one by one.
                    None => mixture(
                        self.lambda / 2.0,
                        |j, w| f_value(self.a, self.b, j, x, q, w),
                        w,
                    ),
                }
            }
        }
    }
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        inverse_result(self.inverse_value(p, upper, left)?, p, true)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        number(finite(
            (self
                .numerator_chi
                .as_ref()
                .map_err(Clone::clone)?
                .sample(&mut r)?
                / self.a)
                / (self.denominator_chi.sample(&mut r)? / self.b),
        )?)
    }
}
statistic_law!(NoncentralF, report);
impl<S: RealScalar> NoncentralF<S> {
    fn inverse_value(&self, p: f64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        if let Some(x) = inverse_endpoint(p, upper, true)? {
            return Ok(x);
        }
        if self.lambda == 0.0 {
            return Work::within(self.options, left, |work| {
                work.charge(1)?;
                f_inverse(self.a, self.b, p, upper, work)
            });
        }
        smooth_inverse(self, true, p, upper, left, |log_p, upper, w| {
            self.initial(log_p, upper, w)
        })
    }
    fn initial(&self, log_p: f64, upper: bool, w: &mut Work) -> Result<f64, SciOxideError> {
        // Patnaik's two-moment scaled-chi-square approximation to the numerator.
        let mean = self.a + self.lambda;
        let degrees = mean * (mean / (self.a + 2.0 * self.lambda));
        Ok((mean / self.a) * f_inverse(degrees, self.b, log_p.exp(), upper, w)?)
    }
}
impl<S: RealScalar> Mean<S> for NoncentralF<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.b <= 2.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        number(finite(
            self.b / (self.b - 2.0) * (1.0 + self.lambda / self.a),
        )?)
    }
}
impl<S: RealScalar> Variance<S> for NoncentralF<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.b <= 4.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        let r = self.lambda / self.a;
        number(finite(
            2.0 * (self.b / (self.b - 2.0)).powi(2)
                * ((1.0 + r).powi(2) + (1.0 + 2.0 * r) * (self.b - 2.0) / self.a)
                / (self.b - 4.0),
        )?)
    }
}

/// Noncentral Student t distribution with `ν > 0` degrees of freedom and noncentrality `δ`.
///
/// `T = (Z + δ) / √(V/ν)` for a standard normal `Z` and an independent
/// chi-squared variate `V` with `ν` degrees of freedom. Tails and densities use
/// the Benton–Krishnamoorthy beta series for `t ≥ 0` and the reflection
/// `T(δ) = −T(−δ)` for `t < 0`. When the series would lose the requested
/// accuracy to cancellation (negative `δ`, far tails), they use quadrature over
/// the chi-squared scale instead. Quantiles start from a normal with matched
/// moments, or a shifted central t when the variance does not exist. They then
/// refine the smaller log tail with a safeguarded Newton iteration in an
/// `asinh` coordinate. `δ = 0` uses the central Student t law.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, NoncentralStudentT, Quantile, SeedableRng};
/// use scioxide_random::SplitMix64;
/// let law = NoncentralStudentT::new(10.0_f64, 1.0)?;
/// // P(T <= 0) = P(Normal(delta, 1) <= 0) = Phi(-1).
/// assert!((law.cdf(0.0)? - 0.158_655_253_931_457_05).abs() < 1e-12);
/// assert!((law.cdf(law.quantile(0.9)?)? - 0.9).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. Benton and K. Krishnamoorthy, "Computing discrete mixtures of continuous
/// distributions: noncentral chisquare, noncentral t and the distribution of the
/// square of the sample multiple correlation coefficient", *Comput. Stat. Data
/// Anal.* 43(2), 2003, 249–267.
#[derive(Debug, Clone)]
pub struct NoncentralStudentT<S: RealScalar> {
    nu: f64,
    delta: f64,
    chi: ChiSquareKernel<f64>,
    /// The central law with default options: zero noncentrality and the
    /// shifted start below two degrees.
    central: Result<crate::StudentT<f64>, SciOxideError>,
    options: LawOptions<S>,
}
impl<S: RealScalar> NoncentralStudentT<S> {
    /// Creates a noncentral t law with `degrees` degrees of freedom and noncentrality
    /// `noncentrality`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if `degrees ≤ 0`.
    pub fn new(degrees: S, noncentrality: S) -> Result<Self, SciOxideError> {
        Self::with_options(degrees, noncentrality, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        degrees: S,
        noncentrality: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(noncentrality, "noncentrality")?;
        let nu = positive(degrees, scioxide_core::argument!("degrees"))?.into_f64();
        Ok(Self {
            nu,
            delta: noncentrality.into_f64(),
            chi: ChiSquareKernel::new(nu)?,
            central: crate::StudentT::new(nu),
            options,
        })
    }
}
impl<S: RealScalar> Statistic<S> for NoncentralStudentT<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn median(&self) -> f64 {
        self.delta
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if self.delta == 0.0 {
            w.charge(1)?;
            let t = self.central.as_ref().map_err(Clone::clone)?;
            return match q {
                Query::Density => t.log_pdf(x),
                Query::Lower => t.log_cdf(x),
                Query::Upper => t.log_sf(x),
            };
        }
        if x == 0.0 && q != Query::Density {
            w.charge(1)?;
            return normal_log(-self.delta, q == Query::Upper);
        }
        if let Some(value) = self.series(x, q, w)? {
            return Ok(value);
        }
        // T = (Z + delta) / s, so P(T <= x) = E[Phi(x s - delta)] over the
        // chi scale s; the density picks up the Jacobian e^r = s.
        scale_integral(
            self.nu,
            |r, _| {
                let z = x * r.exp() - self.delta;
                // A nested Gaussian tail beyond the representable logarithmic
                // range contributes zero to this integral. Do not send its
                // overflowing square through the public normal query contract.
                if z.abs() > f64::MAX.sqrt() {
                    return Ok(if q == Query::Density || (z > 0.0) == (q == Query::Upper) {
                        f64::NEG_INFINITY
                    } else {
                        0.0
                    });
                }

                Ok(if q == Query::Density {
                    math::log_phi(z) + r
                } else {
                    normal_log(z, q == Query::Upper)?
                })
            },
            w,
        )
    }
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        inverse_result(self.inverse_value(p, upper, left)?, p, false)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        number(finite(
            (standard_normal::<f64, _>(&mut r)? + self.delta)
                / (self.chi.sample(&mut r)? / self.nu).sqrt(),
        )?)
    }
}
statistic_law!(NoncentralStudentT, report);
/// Chi-square scale integration in log(sqrt(V/nu)).
pub(super) fn scale_integral(
    nu: f64,
    mut f: impl FnMut(f64, &mut Work) -> Result<f64, SciOxideError>,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    if nu == f64::INFINITY {
        return f(0.0, w);
    }
    // Expectation over the scale s = sqrt(chi^2_nu / nu), integrated in
    // r = ln s. With k = nu/2 its log density is
    // ln 2 + k ln k - ln Gamma(k) - k - k (e^{2r} - 1 - 2r); the bracket is
    // evaluated by `expm1_minus` so the peak at r = 0 has no cancellation.
    // For large k the normalizer is taken from Stirling's series, since
    // k ln k - ln Gamma(k) - k would otherwise cancel almost completely.
    let shape = nu / 2.0;
    let norm = if shape >= 32.0 {
        core::f64::consts::LN_2
            + 0.5 * (shape.ln() - crate::distribution::special::log_two_pi::<f64>())
            - super::super::discrete_math::factorial_stirling_correction(shape)
    } else {
        core::f64::consts::LN_2 + shape * shape.ln() - scioxide_special::ln_gamma(shape)? - shape
    };
    real_integral(
        |r, w| {
            let exp = (2.0 * r).exp();
            if !exp.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            let density = norm - shape * crate::distribution::special::expm1_minus(2.0 * r)?;
            if density == f64::NEG_INFINITY {
                return Ok(density);
            }
            Ok(density + f(r, w)?)
        },
        0.0,
        (1.0 / nu.sqrt()).max(1.0 / nu),
        w,
    )
}
impl<S: RealScalar> NoncentralStudentT<S> {
    /// Benton and Krishnamoorthy (2003): for t >= 0, `F(t) = Phi(-delta) + (A +
    /// sign(delta) B) / 2` and `S(t) = (A' + sign(delta) B') / 2`, where `A` and `B`
    /// are beta mixtures `sum w(j) I_y(j + 1/2, nu/2)` and `sum w(j + 1/2) I_y(j + 1,
    /// nu/2)` (primes: upper tails), `y = t^2 / (nu + t^2)` and `w(k) = e^-m m^k /
    /// Gamma(k + 1)`, `m = delta^2 / 2`; the density is `(A_d + sign(delta) B_d) / t`
    /// from the mixture densities. Negative `t` reflects through `T(delta) = -T(-delta)`.
    /// Positive combinations are accurate to the tolerance; a difference is
    /// summed to a millionth of it and returned only if its cancellation keeps
    /// that accuracy (`None` otherwise, for the quadrature).
    fn series(&self, x: f64, q: Query, w: &mut Work) -> Result<Option<f64>, SciOxideError> {
        if !(x.is_finite() && self.nu.is_finite() && self.nu <= 1.0e12) {
            return Ok(None);
        }
        let (t, delta, q) = if x < 0.0 {
            (
                -x,
                -self.delta,
                match q {
                    Query::Lower => Query::Upper,
                    Query::Upper => Query::Lower,
                    Query::Density => Query::Density,
                },
            )
        } else {
            (x, self.delta, q)
        };
        let (ly, lc) = f_log_coordinates(2.0 * t.ln() - self.nu.ln())?;
        let family = super::mixture::Family::Beta {
            b: 0.5 * self.nu,
            lz: ly,
            ly: lc,
        };
        let mu = 0.5 * delta * delta;
        let integer = super::mixture::Weights { mu, offset: 0.0 };
        let half = super::mixture::Weights { mu, offset: 0.5 };
        let tolerance = w.tolerance;
        let positive = delta >= 0.0;
        // A difference amplifies the truncation of both mixtures.
        if !positive {
            w.tolerance = (tolerance * 1.0e-6).max(64.0 * f64::EPSILON);
        }
        let combined = (|| {
            let (first, second) = match q {
                Query::Density => (
                    Some(super::mixture::log_density(integer, 0.5, family, w)?),
                    Some(super::mixture::log_density(half, 1.0, family, w)?),
                ),
                Query::Lower | Query::Upper => {
                    let upper = q == Query::Upper;
                    (
                        super::mixture::log_tail(integer, 0.5, family, upper, w)?,
                        super::mixture::log_tail(half, 1.0, family, upper, w)?,
                    )
                }
            };
            Ok::<_, SciOxideError>(first.zip(second))
        })();
        let accuracy = w.tolerance;
        w.tolerance = tolerance;
        let Some((first, second)) = combined? else {
            return Ok(None);
        };
        let sum = if positive {
            logadd(first, second)?
        } else if q == Query::Lower {
            // F(t) >= Phi(|delta|) >= 1/2 for t >= 0: no relative loss.
            let value = normal_log(-delta, false)?.exp() + 0.5 * (first.exp() - second.exp());
            return Ok((value > 0.0).then(|| value.ln()));
        } else {
            let gap = second - first;
            // A nonnegative (or undefined) gap leaves no positive difference.
            if gap.is_nan() || gap >= 0.0 {
                return Ok(None);
            }
            let amplification = -(1.0 + gap.exp()) / gap.exp_m1();
            if amplification * (accuracy + 256.0 * f64::EPSILON) > 0.125 * tolerance {
                return Ok(None);
            }
            first + log1mexp(gap)?
        };
        Ok(Some(match q {
            Query::Density => sum - t.ln(),
            Query::Lower => logadd(normal_log(-delta, false)?, sum - core::f64::consts::LN_2)?,
            Query::Upper => sum - core::f64::consts::LN_2,
        }))
    }
    fn inverse_value(&self, p: f64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        if let Some(x) = inverse_endpoint(p, upper, false)? {
            return Ok(x);
        }
        if self.delta == 0.0 {
            crate::law::charge(left)?;
            let law = self.central.as_ref().map_err(Clone::clone)?;
            return if upper { law.isf(p) } else { law.quantile(p) };
        }
        smooth_inverse(self, false, p, upper, left, |log_p, upper, w| {
            self.initial(log_p, upper, w)
        })
    }
    fn initial(&self, log_p: f64, upper: bool, w: &mut Work) -> Result<f64, SciOxideError> {
        let p = log_p.exp();
        if self.nu > 2.0 {
            let z = scioxide_special::normal_quantile(p)?;
            let mean = self.delta * scale_moment_log(self.nu, 1.0)?.exp();
            let log_base = -(-2.0 / self.nu).ln_1p();
            let difference = -(2.0 * scale_moment_log(self.nu, 1.0)? - log_base).exp_m1();
            let variance = log_base.exp() * (1.0 + (self.delta * difference) * self.delta);
            Ok(mean + variance.sqrt() * if upper { -z } else { z })
        } else {
            w.kernel(|left| {
                let law = self.central.as_ref().map_err(Clone::clone)?;
                Ok(self.delta
                    + crate::law::scoped(left, |context| {
                        if upper {
                            law.isf_scoped(p, context)
                        } else {
                            law.quantile_scoped(p, context)
                        }
                    })?)
            })
        }
    }
}
impl<S: RealScalar> Mean<S> for NoncentralStudentT<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.nu <= 1.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        number(finite(self.delta * scale_moment_log(self.nu, 1.0)?.exp())?)
    }
}
impl<S: RealScalar> Variance<S> for NoncentralStudentT<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.nu <= 2.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        // Var = nu/(nu-2) (1 + delta^2) - delta^2 E[1/s]^2, regrouped as
        // nu/(nu-2) (1 + delta^2 (1 - E[1/s]^2 (nu-2)/nu)), where the inner
        // difference is formed with `expm1` because both terms tend to 1 as
        // nu grows.
        let log_base = -(-2.0 / self.nu).ln_1p();
        let difference = -(2.0 * scale_moment_log(self.nu, 1.0)? - log_base).exp_m1();
        let v = log_base.exp() * (1.0 + (self.delta * difference) * self.delta);
        number(crate::distribution::moment_math::positive_variance(v)?)
    }
}

pub(super) fn scale_moment_log(nu: f64, order: f64) -> Result<f64, SciOxideError> {
    if nu <= order {
        return Err(crate::law::undefined_moment(
            "requested moment does not exist",
        ));
    }
    if nu == f64::INFINITY {
        return Ok(0.0);
    }
    crate::distribution::special::log_gamma_ratio_remainder(nu / 2.0, -order / 2.0)
}
