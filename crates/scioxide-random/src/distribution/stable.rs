//! Nolan's positive piecewise stable-law integrals and CMS sampling.
use crate::distribution::LawOptions;
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::constant as number;
use crate::law::{argument, exhausted, finite, log1mexp, logadd};
use crate::{Distribution, Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Standard Lévy alpha-stable distribution with stability `α` and skewness `β`,
/// in Nolan's S1 parameterization.
///
/// The characteristic function is
///
/// ```text
/// φ(t) = exp(−|t|^α (1 − iβ sgn(t) tan(πα/2))),        α ≠ 1,
/// φ(t) = exp(−|t| (1 + iβ (2/π) sgn(t) ln|t|)),         α = 1,
/// ```
///
/// with `0 < α ≤ 2` and `−1 ≤ β ≤ 1`. The law has unit scale and location 0.
/// Shift and scale it with [`Affine`](crate::Affine). `α = 2` is the normal
/// distribution with variance 2, and for `α > 1` the mean is 0. The median of
/// a symmetric law (`β = 0`) is exactly 0. Parameters near `α = 1` are used as
/// given, not rounded to 1.
///
/// The density and distribution functions use Nolan's integral representation,
/// computed by quadrature to the tolerance in [`LawOptions`]. Series expansions
/// replace it near the origin and in light tails where their remainders are
/// negligible. Heavy tails use the Bergström–Zolotarev series while bounds on
/// its terms halve at each step and the first omitted bound is below the
/// tolerance. Quantiles invert the CDF numerically. Sampling uses the
/// Chambers–Mallows–Stuck method.
///
/// The CDF and survival function can round to 0 or 1. Their logarithms are
/// `−∞` only where the probability is exactly 0. A logarithmic density or tail,
/// or an interior quantile, outside the range of `S` is a `NumericalInstability`
/// error.
///
/// # Accuracy
///
/// With the default options, tests check a relative error of at most `8·10⁻⁸`
/// in the smaller tail probability for `f64`, and `8√ε` for `f32`. Below the
/// range of `S`, the logarithmic tails meet the same bound scaled to the
/// logarithm. Quantile accuracy is limited by the tail accuracy and the density
/// at the result. These are tested estimates, not guaranteed bounds.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, LevyStable, Mean, Quantile, SeedableRng, SplitMix64, Variance,
/// };
/// let law = LevyStable::<f64>::new(0.99, 0.5)?;
/// let x = law.quantile(0.75)?;
/// assert!((law.cdf(x)? - 0.75).abs() < 1e-7);
/// assert_eq!(LevyStable::<f64>::new(1.5, 0.)?.quantile(0.5)?, 0.);
/// // S1: for alpha > 1 the mean is the location, zero even under full skew;
/// // alpha = 2 is the normal law with variance 2.
/// let skewed = LevyStable::<f64>::new(1.5, 1.)?;
/// assert_eq!(skewed.mean()?, 0.);
/// assert_eq!(LevyStable::<f64>::new(2., 0.)?.variance()?, 2.);
/// let mut rng = SplitMix64::from_seed(66);
/// assert!(skewed.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. P. Nolan, "Numerical calculation of stable densities and distribution
///   functions", *Commun. Statist. Stochastic Models* 13(4), 1997, 759–774.
/// - J. M. Chambers, C. L. Mallows and B. W. Stuck, "A method for simulating
///   stable random variables", *J. Amer. Statist. Assoc.* 71(354), 1976,
///   340–344.
#[derive(Debug, Clone, Copy)]
pub struct LevyStable<S: RealScalar> {
    alpha: S,
    beta: S,
    half_gamma: crate::distribution::gamma::GammaKernel<S>,
    options: LawOptions<S>,
}
impl<S: RealScalar> LevyStable<S> {
    /// Creates a standard stable distribution with stability `alpha` and
    /// skewness `beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `DomainError` if `alpha ∉ (0, 2]` or `|beta| > 1`.
    pub fn new(alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(alpha, beta, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(alpha: S, beta: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(alpha, "alpha")?;
        argument(beta, "beta")?;
        if alpha <= S::ZERO || alpha > number::<S>(2.)? || beta.abs() > S::ONE {
            return Err(crate::law::domain(
                if beta.abs() > S::ONE {
                    scioxide_core::argument!("beta")
                } else {
                    scioxide_core::argument!("alpha")
                },
                "stable law requires 0<alpha<=2 and |beta|<=1",
            ));
        }
        Ok(Self {
            alpha,
            beta,
            half_gamma: crate::distribution::gamma::GammaKernel::new(S::HALF)?,
            options,
        })
    }
    /// Returns the options used by every query.
    pub const fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn bounds_f64(&self) -> (Option<f64>, Option<f64>) {
        if self.alpha.into_f64() < 1. && self.beta == S::ONE {
            (Some(0.), None)
        } else if self.alpha.into_f64() < 1. && self.beta == -S::ONE {
            (None, Some(0.))
        } else {
            (None, None)
        }
    }
    fn numeric(&self, x: S, kind: IntegralKind, left: &mut usize) -> Result<S, SciOxideError> {
        let a = self.alpha.into_f64();
        let b = self.beta.into_f64();
        let x = x.into_f64();
        let tol = self.options.tolerance.relative().into_f64();
        let v = stable_log(a, b, x, kind, tol, left, None)?;
        if v == f64::NEG_INFINITY {
            let support_zero = a < 1. && ((b == 1. && x <= 0.) || (b == -1. && x >= 0.));
            if matches!(kind, IntegralKind::Density) && !support_zero {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "stable log density is outside the finite scalar range",
                ));
            }
            Ok(S::NEG_INFINITY)
        } else {
            S::try_from_f64_lossy(v)
        }
    }

    fn overflowing_log_tail(&self, x: S) -> Option<bool> {
        let (a, mut b, mut x) = (self.alpha.into_f64(), self.beta.into_f64(), x.into_f64());
        if a == 2. {
            return (x.abs() / 2. > S::MAX.into_f64().sqrt()).then_some(x > 0.);
        }
        if a < 1. && b.abs() == 1. && b * x <= 0. {
            return None;
        }
        let reflected = (a != 1. && x < 0.) || (a == 1. && b < 0.);
        if reflected {
            b = -b;
            x = -x;
        }
        Nolan::new(a, b, x, None)
            .log_tail_overflows(S::MAX.into_f64())
            .then_some((a > 1.) != reflected)
    }

    fn inverse_probability(
        &self,
        p: S,
        upper: bool,
        remaining: &mut usize,
    ) -> Result<S, SciOxideError> {
        use crate::{InverseSurvival, Quantile};
        use core::cell::RefCell;
        use scioxide_core::{ResidualTolerance, StepTolerance};
        use scioxide_rootfind::{Brent, ScalarRootOptions};
        if p == S::ZERO || p == S::ONE {
            return crate::distribution::formula::numerical_inverse(
                self, p, upper, None, remaining,
            );
        }
        let a = self.alpha.into_f64();
        let b = self.beta.into_f64();
        let p = p.into_f64();
        if a == 2. {
            let law = crate::Normal::new(0., 2_f64.sqrt())?;
            return S::try_from_f64_lossy(if upper { law.isf(p)? } else { law.quantile(p)? });
        }
        if a == 1. && b == 0. {
            let law = crate::Cauchy::new(0., 1.)?;
            return S::try_from_f64_lossy(if upper { law.isf(p)? } else { law.quantile(p)? });
        }
        if a == 0.5 && b.abs() == 1. {
            let t = if upper != (b < 0.) {
                scioxide_special::erfinv(p)?
            } else {
                scioxide_special::erfcinv(p)?
            };
            return S::try_from_f64_lossy(b * (0.5 / t) / t);
        }
        let tol = self.options.tolerance.relative().into_f64();
        if b == 0. && (p - 0.5).abs() < 0.001 {
            let log_f0 = scioxide_special::ln_gamma(1. + 1. / a)? - core::f64::consts::PI.ln();
            let x = (p - 0.5) * (-log_f0).exp();
            if x == 0. {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "stable interior quantile is outside the finite scalar range",
                ));
            }
            let log_relative_bound = crate::distribution::special::log_gamma_ratio(1. / a, 2. / a)?
                - 6_f64.ln()
                + 2. * x.abs().ln();
            if log_relative_bound < (tol / 16.).ln() {
                return S::try_from_f64_lossy(if upper { -x } else { x });
            }
        }
        let small = p.min(1. - p);
        let upper = upper != (p > 0.5);
        let target = small.ln();
        let centered = (a - 1.).abs() < 0.05 && b != 0.;
        let (shift, low) = if centered {
            nolan::shift(a, b)
        } else {
            (0., 0.)
        };
        // Small-alpha laws span exponentially many scales; invert log|x|
        // on the appropriate side of zero instead of losing tiny quantiles
        // to an absolute root tolerance or a location subtraction.
        let log_sign = if a < 0.5 {
            let origin = Nolan::new(a, b, 0., None);
            let at_zero = if upper {
                origin.origin_upper()
            } else {
                origin.origin_lower()
            };
            if small == at_zero {
                return Ok(S::ZERO);
            }
            Some(if (small < at_zero) == upper { 1. } else { -1. })
        } else {
            None
        };
        let coordinate = |y: f64| log_sign.map_or(shift + (low + y), |sign| sign * y.exp());
        let left = RefCell::new(remaining);
        let residual = |y: f64| {
            let mut remaining = left.borrow_mut();
            nolan::charge(&mut remaining)?;
            let value = stable_log(
                a,
                b,
                coordinate(y),
                if upper {
                    IntegralKind::Upper
                } else {
                    IntegralKind::Lower
                },
                tol * 0.125,
                &mut remaining,
                centered.then_some(y),
            );
            // A bounded continuous residual also covers exact support zeros.
            let r = (value? - target).exp_m1();
            Ok(log_sign.unwrap_or(1.)
                * if upper {
                    -r.min(1. / small)
                } else {
                    r.min(1. / small)
                })
        };
        let lower_support = if log_sign.is_none() && a < 1. && b == 1. {
            Some(-shift - low)
        } else {
            None
        };
        let upper_support = if log_sign.is_none() && a < 1. && b == -1. {
            Some(-shift - low)
        } else {
            None
        };
        let minimum = if log_sign.is_some() {
            f64::from_bits(1).ln()
        } else {
            -f64::MAX
        };
        let maximum = if log_sign.is_some() {
            f64::MAX.ln()
        } else {
            f64::MAX
        };
        let options = ScalarRootOptions::default()
            .with_max_iterations(self.options.work)
            .with_residual_tolerance(ResidualTolerance::new(tol * 0.25, 0.)?)
            .with_step_tolerance(StepTolerance::new(0., 4. * f64::EPSILON)?);
        let result = crate::inversion::solve_monotone(
            residual,
            (
                lower_support.map_or(-1., |v| v.max(-1.)),
                upper_support.map_or(1., |v| v.min(1.)),
            ),
            (
                lower_support.unwrap_or(minimum),
                upper_support.unwrap_or(maximum),
            ),
            &options,
            Brent,
        )?;
        if !result.is_success() {
            return Err(exhausted());
        }
        let x = S::try_from_f64_lossy(coordinate(result.point()))?;
        if (log_sign.is_some() || (self.alpha < S::ONE && self.beta.abs() == S::ONE))
            && x == S::ZERO
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "stable interior quantile rounds to its support endpoint",
            ));
        }
        finite(x)
    }
}

#[derive(Clone, Copy)]
enum IntegralKind {
    Density,
    Lower,
    Upper,
}
mod nolan;
use nolan::Nolan;

fn stable_log(
    alpha: f64,
    beta: f64,
    x: f64,
    kind: IntegralKind,
    tol: f64,
    limit: &mut usize,
    centered: Option<f64>,
) -> Result<f64, SciOxideError> {
    let pi = core::f64::consts::PI;
    if alpha == 2. {
        let z = x / 2.;
        return Ok(match kind {
            IntegralKind::Density => -z * z - (4. * pi).ln() * 0.5,
            IntegralKind::Lower => scioxide_special::normal_log_cdf(x / 2f64.sqrt())?,
            IntegralKind::Upper => scioxide_special::normal_log_sf(x / 2f64.sqrt())?,
        });
    }
    if alpha == 1. && beta == 0. {
        let d = crate::Cauchy::new(0., 1.)?;
        return match kind {
            IntegralKind::Density => crate::LogDensity::log_pdf(&d, x),
            IntegralKind::Lower => crate::LogCdf::log_cdf(&d, x),
            IntegralKind::Upper => crate::LogSurvival::log_sf(&d, x),
        };
    }
    if beta == 0. && (alpha - 1.).abs() < 0.001 && x.abs() <= 1e6 {
        // First alpha derivative of the characteristic-function integral at
        // alpha=1. This retains the requested alpha when the angular peak is
        // narrower than an IEEE coordinate. The second-order absolute
        // remainder is bounded conservatively by 64*(alpha-1)^2 here.
        let d = alpha - 1.;
        let u = x.abs();
        let denominator = 1. + u * u;
        let angle = u.atan();
        let gamma = 0.577_215_664_901_532_9;
        let log_radius = denominator.ln() * 0.5;
        let small = if u == 0. { 0.5 } else { (1. / u).atan() / pi };
        let derivative = (u * (gamma + log_radius) - angle) / (pi * denominator);
        let density = 1. / (pi * denominator);
        let target = if matches!(kind, IntegralKind::Density) {
            density
        } else {
            small
        };
        if 64. * d * d < tol * target / 16. {
            nolan::charge(limit)?;
            let tail = small - d * derivative;
            return Ok(match kind {
                IntegralKind::Density => (density
                    - d * ((1. - gamma - log_radius) * (1. - u * u) - 2. * u * angle)
                        / (pi * denominator * denominator))
                    .ln(),
                IntegralKind::Lower if x < 0. => tail.ln(),
                IntegralKind::Upper if x >= 0. => tail.ln(),
                _ => (-tail).ln_1p(),
            });
        }
    }
    if alpha == 0.5 && beta.abs() == 1. {
        let y = beta * x;
        if y <= 0. {
            return Ok(match kind {
                IntegralKind::Density => f64::NEG_INFINITY,
                IntegralKind::Lower => {
                    if beta > 0. {
                        f64::NEG_INFINITY
                    } else {
                        0.
                    }
                }
                IntegralKind::Upper => {
                    if beta > 0. {
                        0.
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            });
        }
        let t = (0.5 / y).sqrt();
        return Ok(match kind {
            IntegralKind::Density => -(2. * pi).ln() * 0.5 - 1.5 * y.ln() - 0.5 / y,
            IntegralKind::Lower => {
                if beta > 0. {
                    scioxide_special::log_erfc(t)?
                } else {
                    scioxide_special::erf(t)?.ln()
                }
            }
            IntegralKind::Upper => {
                if beta > 0. {
                    scioxide_special::erf(t)?.ln()
                } else {
                    scioxide_special::log_erfc(t)?
                }
            }
        });
    }
    if alpha < 1. && beta == 1. && x <= 0. {
        return Ok(match kind {
            IntegralKind::Density | IntegralKind::Lower => f64::NEG_INFINITY,
            IntegralKind::Upper => 0.,
        });
    }
    if alpha < 1. && beta == -1. && x >= 0. {
        return Ok(match kind {
            IntegralKind::Density | IntegralKind::Upper => f64::NEG_INFINITY,
            IntegralKind::Lower => 0.,
        });
    }
    if (alpha != 1. && x < 0.) || (alpha == 1. && beta < 0.) {
        return stable_log(
            alpha,
            -beta,
            -x,
            match kind {
                IntegralKind::Density => IntegralKind::Density,
                IntegralKind::Lower => IntegralKind::Upper,
                IntegralKind::Upper => IntegralKind::Lower,
            },
            tol,
            limit,
            centered.map(|y| -y),
        );
    }
    if beta > -1. && x > 0. {
        if alpha != 1. {
            if let Some(value) = tail_series(alpha, beta, x, kind, tol)? {
                nolan::charge(limit)?;
                return Ok(value);
            }
        } else if 32. * (1. + x.ln().abs()) / x < tol / 16. {
            // Leading alpha=1 tail; the next term includes log(x)/x.
            nolan::charge(limit)?;
            let log_upper = (1. + beta).ln() - pi.ln() - x.ln();
            return Ok(match kind {
                IntegralKind::Density => log_upper - x.ln(),
                IntegralKind::Lower => log1mexp(log_upper)?,
                IntegralKind::Upper => log_upper,
            });
        }
    }
    let n = Nolan::new(alpha, beta, x, centered);
    if alpha != 1. && x == 0. {
        return match kind {
            IntegralKind::Density => Ok(scioxide_special::ln_gamma(1. + 1. / alpha)?
                + scioxide_core::sin_pi(n.origin_lower())?.ln()
                - pi.ln()
                - (1. + n.zeta * n.zeta).ln() / (2. * alpha)),
            IntegralKind::Lower => Ok(n.origin_lower().max(0.).ln()),
            IntegralKind::Upper => Ok(n.origin_upper().max(0.).ln()),
        };
    }
    if let Some(value) = n.origin(kind, tol, limit)? {
        return Ok(value);
    }
    let direct = match kind {
        IntegralKind::Density => IntegralKind::Density,
        IntegralKind::Lower if alpha > 1. => IntegralKind::Upper,
        IntegralKind::Upper if alpha > 1. => IntegralKind::Lower,
        _ => kind,
    };
    let logarithm = n.log_integral(direct, tol, limit)?;
    match kind {
        IntegralKind::Density => Ok(logarithm
            + if alpha == 1. {
                (0.5 / beta.abs()).ln()
            } else {
                (alpha / (pi * (alpha - 1.).abs() * x)).ln()
            }),
        IntegralKind::Upper => Ok(logarithm - pi.ln()),
        IntegralKind::Lower if alpha == 1. => Ok(logarithm - pi.ln()),
        IntegralKind::Lower => logadd(n.origin_lower().max(0.).ln(), logarithm - pi.ln()),
    }
}

/// Bergström–Zolotarev tail series for `x > 0`, `alpha != 1` and `beta > -1`:
/// `P(X > x) = Σ (-1)^(k+1) Γ(kα) Im(w^k) x^(-kα) / (π k!)` with
/// `w = exp(iπα/2) (1 + i β tan(πα/2))`, and the density with `Γ(kα+1)` and
/// one more power of `1/x`. The series converges for `alpha < 1` and is
/// asymptotic for `alpha > 1`. It is used only while bounds `|Im(w^k)| <= |w|^k`
/// on its terms halve at each step and the first omitted bound is below a
/// thirty-second of the tolerance relative to the partial sum.
fn tail_series(
    alpha: f64,
    beta: f64,
    x: f64,
    kind: IntegralKind,
    tol: f64,
) -> Result<Option<f64>, SciOxideError> {
    const TERMS: usize = 24;
    let (sin, cos) = scioxide_core::sin_cos_pi(alpha * 0.5)?;
    let q = nolan::shift(alpha, beta).0;
    let w = (cos - q * sin, sin * (1. + beta));
    let modulus = w.0.hypot(w.1);
    let unit = (w.0 / modulus, w.1 / modulus);
    let density = matches!(kind, IntegralKind::Density);
    let offset = if density { 1. } else { 0. };
    let log_gamma = |k: f64| scioxide_special::ln_gamma(k * alpha + offset);
    // Term k relative to term 1, without the Im(u^k) factor.
    let step = modulus.ln() - alpha * x.ln();
    let first = log_gamma(1.)?;
    let mut power = unit;
    let mut sum = unit.1;
    let mut previous = 0.;
    for k in 2..=TERMS {
        let kf = k as f64;
        let log_bound =
            log_gamma(kf)? - first - scioxide_special::ln_gamma(kf + 1.)? + (kf - 1.) * step;
        if log_bound - previous > -core::f64::consts::LN_2 {
            return Ok(None);
        }
        // Both the next bound and the geometric remainder after it.
        if sum > 0. && 2. * log_bound.exp() <= tol / 32. * sum {
            let log_value =
                first + modulus.ln() - alpha * x.ln() + sum.ln() - core::f64::consts::PI.ln();
            return Ok(Some(match kind {
                IntegralKind::Density => log_value - x.ln(),
                IntegralKind::Lower => log1mexp(log_value)?,
                IntegralKind::Upper => log_value,
            }));
        }
        power = (
            power.0 * unit.0 - power.1 * unit.1,
            power.0 * unit.1 + power.1 * unit.0,
        );
        let sign = if k % 2 == 0 { -1. } else { 1. };
        sum += sign * log_bound.exp() * power.1;
        previous = log_bound;
    }
    Ok(None)
}

impl<S: RealScalar> Formula<S> for LevyStable<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        let (a, b) = self.bounds_f64();
        (a.map(|_| S::ZERO), b.map(|_| S::ZERO))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        self.numeric(
            x,
            IntegralKind::Density,
            &mut self.options.work_budget().get(),
        )
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            self.numeric(x, IntegralKind::Upper, _left)
        } else {
            self.numeric(x, IntegralKind::Lower, _left)
        }
    }
    fn log_probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        self.log_probability_with_work(x, upper, &mut self.options.work_budget().get())
    }
    fn log_probability_with_work(
        &self,
        x: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if let Some(small_upper) = self.overflowing_log_tail(x) {
            return if upper != small_upper {
                Ok(S::ZERO)
            } else {
                Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "stable log tail is outside the finite scalar range",
                ))
            };
        }
        let (a, b, x) = (self.alpha.into_f64(), self.beta.into_f64(), x.into_f64());
        let tol = self.options.tolerance.relative().into_f64();
        // The S1 shift locates the mode; the tail beyond it is evaluated first.
        let (value, _) =
            crate::law::smaller_tail_first(x >= nolan::shift(a, b).0, upper, |upper| {
                let kind = if upper {
                    IntegralKind::Upper
                } else {
                    IntegralKind::Lower
                };
                stable_log(a, b, x, kind, tol, left, None)
            })?;
        if value == f64::NEG_INFINITY {
            Ok(S::NEG_INFINITY)
        } else {
            S::try_from_f64_lossy(value)
        }
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if let Some(small_upper) = self.overflowing_log_tail(x) {
            return Ok(if upper == small_upper {
                S::ZERO
            } else {
                S::ONE
            });
        }
        Ok(self.log_probability(x, upper)?.exp())
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if self.beta == S::ZERO && p == S::HALF {
            return Ok(S::ZERO);
        }
        self.inverse_probability(p, false, &mut self.options.work_budget().get())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if self.beta == S::ZERO && p == S::HALF {
            return Ok(S::ZERO);
        }
        self.inverse_probability(p, true, &mut self.options.work_budget().get())
    }

    fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.numeric(x, IntegralKind::Density, left)
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if self.beta == S::ZERO && p == S::HALF {
            return Ok(S::ZERO);
        }
        self.inverse_probability(p, upper, left)
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if self.alpha == number::<S>(2.)? {
            let z = crate::distribution::normal::standard_normal::<S, R>(rng)?;
            return Ok(number::<S>(2.)?.sqrt() * z);
        }
        if self.alpha == S::ONE && self.beta == S::ZERO {
            return crate::Cauchy::new(S::ZERO, S::ONE)?.sample(rng);
        }
        if self.alpha == S::HALF && self.beta.abs() == S::ONE {
            // GeneralizedGamma(1/2, -1): the reciprocal of a Gamma(1/2) variate.
            let base = S::ONE / self.half_gamma.sample(rng)?;
            return finite(self.beta * S::HALF * base);
        }
        // Chambers–Mallows–Stuck in binary64: a binary32 angle next to ±pi/2
        // would round cos(u - a u) through zero. The construction divides by
        // the exponential variate, so a ziggurat zero is a rejected proposal.
        let pi = core::f64::consts::PI;
        let u = pi * (crate::util::validated_open_float(rng)?.into_f64() - 0.5);
        let w = crate::util::rejection(|| {
            let w = super::exponential::standard_exponential::<S, R>(rng)?.into_f64();
            Ok((w > 0.0).then_some(w))
        })?;
        let (a, b) = (self.alpha.into_f64(), self.beta.into_f64());
        let x = if a == 1.0 {
            let h = pi * 0.5;
            let bu = b * u;
            (2.0 / pi) * ((h + bu) * u.tan() - b * ((h * w * u.cos()) / (h + bu)).ln())
        } else {
            let q = b * (pi * a * 0.5).tan();
            let angle = q.atan() / a;
            let scale = (1.0 + q * q).powf(1.0 / (2.0 * a));
            let au = a * (u + angle);
            scale * au.sin() / u.cos().powf(1.0 / a) * ((u - au).cos() / w).powf((1.0 - a) / a)
        };
        finite(number::<S>(finite(x)?)?)
    }
}
formula_law!(LevyStable);
impl<S: RealScalar> Mean<S> for LevyStable<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.alpha > S::ONE {
            Ok(S::ZERO)
        } else {
            Err(crate::law::undefined_moment(
                "stable-law mean does not exist",
            ))
        }
    }
}
impl<S: RealScalar> Variance<S> for LevyStable<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.alpha == number::<S>(2.)? {
            Ok(number(2.)?)
        } else {
            Err(crate::law::undefined_moment(
                "stable-law variance does not exist",
            ))
        }
    }
}
