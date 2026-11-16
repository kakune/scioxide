//! Distributions on bounded intervals with elementary densities.
use crate::distribution::formula::formula_law;
use crate::distribution::formula::{Formula, Inverse};
use crate::distribution::moment_math::variance_from_moments;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{argument, finite, log1mexp};
use crate::{Mean, Variance};
use scioxide_core::{RealScalar, SciOxideError};

pub(super) fn unit<S: RealScalar>() -> (Option<S>, Option<S>) {
    (Some(S::ZERO), Some(S::ONE))
}

/// Anglit distribution on `[−π/4, π/4]` with density `f(x) = cos(2x)`.
///
/// ```
/// use scioxide_random::{
///     Anglit, Cdf, Distribution, SeedableRng, SplitMix64, Support, Variance,
/// };
/// use core::f64::consts::PI;
/// let law = Anglit::<f64>::new();
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-8);
/// assert!((law.variance()? - (PI * PI / 16.0 - 0.5)).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(31);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Anglit<S: RealScalar>(core::marker::PhantomData<S>);
impl<S: RealScalar> Anglit<S> {
    /// Creates the standard anglit distribution.
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
impl<S: RealScalar> Default for Anglit<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> Formula<S> for Anglit<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        let four = S::TWO + S::TWO;
        (Some(-S::PI / four), Some(S::PI / four))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let q = S::PI / number::<S>(4.)?;
        Ok(if x.abs() >= q {
            S::NEG_INFINITY
        } else {
            (number::<S>(2.)? * x).cos().ln()
        })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        let x = if upper { -x } else { x };
        let q = S::PI / number::<S>(4.)?;
        Ok(if x <= -q {
            S::NEG_INFINITY
        } else if x >= q {
            S::ZERO
        } else {
            number::<S>(2.)? * (x + q).sin().ln()
        })
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        finite(p.sqrt().asin() - S::PI / number::<S>(4.)?)
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        Ok(-self.inverse(p)?)
    }
}
formula_law!(Anglit);
impl<S: RealScalar> Mean<S> for Anglit<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for Anglit<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(S::PI * S::PI / number::<S>(16.)? - S::HALF)
    }
}

/// Cosine distribution on `[−π, π]` with density `f(x) = (1 + cos x) / (2π)`.
///
/// Quantiles are computed by numerical inversion of the CDF.
///
/// ```
/// use scioxide_random::{
///     Cdf, Cosine, Distribution, Quantile, SeedableRng, SplitMix64, Variance,
/// };
/// use core::f64::consts::PI;
/// let law = Cosine::<f64>::new()?;
/// let x = law.quantile(0.9)?;
/// assert!((law.cdf(x)? - 0.9).abs() < 1e-8);
/// assert!((law.variance()? - (PI * PI / 3.0 - 2.0)).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(32);
/// assert!(law.sample(&mut rng)?.abs() <= PI);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Cosine<S: RealScalar> {
    options: crate::LawOptions<S>,
}
impl<S: RealScalar> Cosine<S> {
    /// Creates the standard cosine distribution.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// None: the default options are valid. The `Result` matches
    /// [`Self::with_options`].
    pub fn new() -> Result<Self, SciOxideError> {
        Self::with_options(crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self { options })
    }
    // F(-pi + t) = (t - sin t) / (2 pi). For small t the difference cancels
    // catastrophically, so the first two Taylor terms t^3/6 - t^5/120 are
    // used below 0.01, where the next term is below double rounding.
    fn small_tail(t: S) -> Result<S, SciOxideError> {
        let t3 = t * t * t;
        Ok(if t < number(0.01)? {
            t3 * (number::<S>(1. / 6.)? - t * t * number::<S>(1. / 120.)?)
        } else {
            t - t.sin()
        })
    }
}
impl<S: RealScalar> Formula<S> for Cosine<S> {
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
        (Some(-S::PI), Some(S::PI))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x.abs() >= S::PI {
            return Ok(S::NEG_INFINITY);
        }
        let a = (x * S::HALF).cos();
        Ok(number::<S>(2.)? * a.ln() - S::PI.ln())
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        let x = if upper { -x } else { x };
        let p = S::PI;
        if x <= -p {
            return Ok(S::NEG_INFINITY);
        }
        if x >= p {
            return Ok(S::ZERO);
        }
        // Always evaluate the tail nearer its support end and complement for
        // the other half, so both tails stay accurate.
        if x <= S::ZERO {
            Ok((Self::small_tail(x + p)? / (number::<S>(2.)? * p)).ln())
        } else {
            log1mexp((Self::small_tail(p - x)? / (number::<S>(2.)? * p)).ln())
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;
}
formula_law!(Cosine);
impl<S: RealScalar> Mean<S> for Cosine<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for Cosine<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let p = S::PI;
        finite(p * p / number::<S>(3.)? - number::<S>(2.)?)
    }
}

/// Bradford distribution on `[0, 1]` with shape `c > 0`.
///
/// `F(x) = ln(1 + cx) / ln(1 + c)`.
///
/// ```
/// use scioxide_random::{
///     Bradford, Cdf, Distribution, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = Bradford::<f64>::new(2.0)?;
/// assert!((law.cdf(0.5)? - 2.0_f64.ln() / 3.0_f64.ln()).abs() < 1e-8);
/// let x = law.quantile(0.3)?;
/// assert!((law.cdf(x)? - 0.3).abs() < 1e-8);
/// assert!((law.mean()? - (2.0 - 3.0_f64.ln()) / (2.0 * 3.0_f64.ln())).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(33);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Bradford<S: RealScalar> {
    shape: S,
}
impl<S: RealScalar> Bradford<S> {
    /// Creates a Bradford distribution with shape `c = shape`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self { shape })
    }
}
impl<S: RealScalar> Formula<S> for Bradford<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        unit()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(if x < S::ZERO || x > S::ONE {
            S::NEG_INFINITY
        } else {
            self.shape.ln() - self.shape.ln_1p().ln() - (self.shape * x).ln_1p()
        })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x <= S::ZERO {
                S::ZERO
            } else if x >= S::ONE {
                S::NEG_INFINITY
            } else {
                // 1 - ln(1 + c x) / ln(1 + c) = ln((1 + c) / (1 + c x)) / ln(1 + c),
                // rewritten as ln_1p(c (1 - x) / (1 + c x)) so the upper tail near
                // x = 1 is not a difference of nearly equal logarithms.
                (self.shape * (S::ONE - x) / (S::ONE + self.shape * x))
                    .ln_1p()
                    .ln()
                    - self.shape.ln_1p().ln()
            })
        } else {
            Ok(if x <= S::ZERO {
                S::NEG_INFINITY
            } else if x >= S::ONE {
                S::ZERO
            } else {
                (self.shape * x).ln_1p().ln() - self.shape.ln_1p().ln()
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        finite((p * self.shape.ln_1p()).exp_m1() / self.shape)
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ONE);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        finite(((S::ONE - p) * self.shape.ln_1p()).exp_m1() / self.shape)
    }
}
formula_law!(Bradford);
impl<S: RealScalar> Mean<S> for Bradford<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(super::moment_math::bradford(self.shape.into_f64()).0)
    }
}
impl<S: RealScalar> Variance<S> for Bradford<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        super::moment_math::variance(super::moment_math::bradford(self.shape.into_f64()).1.ln())
    }
}

/// Trapezoidal distribution on `[0, 1]` with plateau `[c, d]`.
///
/// The density rises linearly on `[0, c]`, is constant on `[c, d]` and falls
/// linearly on `[d, 1]`. `c = d` gives a triangular law.
///
/// ```
/// use scioxide_random::{Density, Distribution, Mean, SeedableRng, SplitMix64, Trapezoid};
/// let law = Trapezoid::<f64>::new(0.25, 0.75)?;
/// assert!((law.pdf(0.5)? - 4.0 / 3.0).abs() < 1e-15);
/// assert!((law.mean()? - 0.5).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(34);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Trapezoid<S: RealScalar> {
    c: S,
    d: S,
}
impl<S: RealScalar> Trapezoid<S> {
    /// Creates a trapezoidal distribution with plateau `[c, d]`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` or `d` is NaN or infinite.
    /// - `DomainError` unless `0 ≤ c ≤ d ≤ 1`.
    pub fn new(c: S, d: S) -> Result<Self, SciOxideError> {
        argument(c, "c")?;
        argument(d, "d")?;
        if c < S::ZERO || c > d || d > S::ONE {
            return Err(crate::law::domain(
                if c < S::ZERO || c > d {
                    scioxide_core::argument!("c")
                } else {
                    scioxide_core::argument!("d")
                },
                "trapezoid requires 0 <= c <= d <= 1",
            ));
        }
        Ok(Self { c, d })
    }
    fn width(&self) -> S {
        S::ONE + self.d - self.c
    }
    fn raw_moment(&self, n: u32) -> Result<S, SciOxideError> {
        let power = n as i32 + 2;
        // E[X^n] = 2 / ((1 + d - c)(n + 1)(n + 2)) * ((1 - d^(n+2)) / (1 - d)
        // - c^(n+1)). The geometric quotient is expanded as sum_j d^j so the
        // d -> 1 (triangular right edge) case needs no special branch.
        let tail = (0..power).fold(S::ZERO, |sum, j| sum + self.d.powi(j));
        let c = self.c.powi(n as i32 + 1);
        finite(
            number::<S>(2.)? * (tail - c)
                / (self.width() * number::<S>(((n + 1) * (n + 2)) as f64)?),
        )
    }
}
impl<S: RealScalar> Formula<S> for Trapezoid<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        unit()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO || x > S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        let u = number::<S>(2.)? / self.width();
        Ok(if x < self.c {
            (u * x / self.c).ln()
        } else if x <= self.d {
            u.ln()
        } else {
            (u * (S::ONE - x) / (S::ONE - self.d)).ln()
        })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            if x >= S::ONE {
                return Ok(S::NEG_INFINITY);
            }
            let w = self.width();
            Ok(if x > self.d {
                ((S::ONE - x).powi(2) / (w * (S::ONE - self.d))).ln()
            } else if x >= self.c {
                ((w - number::<S>(2.)? * x + self.c) / w).ln()
            } else {
                log1mexp((x * x / (self.c * w)).ln())?
            })
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            if x >= S::ONE {
                return Ok(S::ZERO);
            }
            let w = self.width();
            Ok(if x < self.c {
                (x * x / (self.c * w)).ln()
            } else if x <= self.d {
                ((number::<S>(2.)? * x - self.c) / w).ln()
            } else {
                log1mexp(((S::ONE - x).powi(2) / (w * (S::ONE - self.d))).ln())?
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        // Piecewise inversion of the rising, flat and falling pieces; pc and pd
        // are the CDF values at the breakpoints c and d.
        let w = self.width();
        let pc = self.c / w;
        let pd = (number::<S>(2.)? * self.d - self.c) / w;
        finite(if p < pc {
            p.sqrt() * (self.c * w).sqrt()
        } else if p <= pd {
            (p * w + self.c) * S::HALF
        } else {
            S::ONE - (S::ONE - p).sqrt() * (w * (S::ONE - self.d)).sqrt()
        })
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        let w = self.width();
        let qd = (S::ONE - self.d) / w;
        let qc = S::ONE - self.c / w;
        finite(if p < qd {
            S::ONE - p.sqrt() * (w * (S::ONE - self.d)).sqrt()
        } else if p <= qc {
            (w * (S::ONE - p) + self.c) * S::HALF
        } else {
            (S::ONE - p).sqrt() * (self.c * w).sqrt()
        })
    }
}
formula_law!(Trapezoid);
impl<S: RealScalar> Mean<S> for Trapezoid<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(1)
    }
}
impl<S: RealScalar> Variance<S> for Trapezoid<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        variance_from_moments(self.raw_moment(2)?, m)
    }
}

/// ARGUS distribution on `[0, 1]` with shape `χ > 0`.
///
/// The density is `f(x) ∝ x √(1 − x²) exp(−χ² (1 − x²) / 2)`. Its normalizing
/// constant is proportional to the regularized incomplete gamma function
/// `P(3/2, χ²/2)` (DLMF 8.2.4).
///
/// ```
/// use scioxide_random::{
///     Argus, Cdf, Distribution, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = Argus::<f64>::new(1.0)?;
/// let x = law.quantile(0.5)?;
/// assert!((law.cdf(x)? - 0.5).abs() < 1e-13);
/// assert!(law.mean()? > 0.0 && law.mean()? < 1.0);
/// let mut rng = SplitMix64::from_seed(35);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Argus<S: RealScalar> {
    options: crate::LawOptions<S>,
    chi: S,
    a: S,
    log_norm: S,
}
impl<S: RealScalar> Argus<S> {
    /// Creates an ARGUS distribution with shape `χ = chi`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `chi` is NaN or infinite.
    /// - `DomainError` if `chi ≤ 0`.
    /// - `NumericalInstability` if `χ²/2` overflows or the normalizing constant underflows.
    pub fn new(chi: S) -> Result<Self, SciOxideError> {
        Self::with_options(chi, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(chi: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(chi, scioxide_core::argument!("chi"))?;
        let left = &mut options.work_budget().get();
        let a = chi * chi * S::HALF;
        if !a.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "squared shape exceeds scalar range",
            ));
        }
        // For chi^2 / 2 below eps the law equals its chi -> 0 limit to working
        // precision; switching to the closed-form limit avoids dividing by a
        // normalizer P(3/2, a) that underflows like a^{3/2}.
        let a = if a < S::EPSILON { S::ZERO } else { a };
        let log_norm = if a == S::ZERO {
            S::ZERO
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::gamma_log_p_with_options(
                    number::<S>(1.5)?,
                    a,
                    scioxide_special::GammaOptions::new(budget),
                )
            })?
        };
        if !log_norm.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "shape normalization is not representable",
            ));
        }
        Ok(Self {
            options,
            chi,
            a,
            log_norm,
        })
    }
    fn second_raw(&self, left: &mut usize) -> Result<S, SciOxideError> {
        if self.a == S::ZERO {
            return number(0.4);
        }
        let ratio = (number::<S>(1.5)?.ln()
            + crate::law::special(left, |budget| {
                scioxide_special::gamma_log_p_with_options(
                    number::<S>(2.5)?,
                    self.a,
                    scioxide_special::GammaOptions::new(budget),
                )
            })?
            - self.a.ln()
            - self.log_norm)
            .exp();
        finite(S::ONE - ratio)
    }
}
impl<S: RealScalar> Formula<S> for Argus<S> {
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
        unit()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO || x >= S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        if self.a == S::ZERO {
            return Ok(number::<S>(3.)?.ln() + x.ln() + S::HALF * (-x * x).ln_1p());
        }
        Ok(
            number::<S>(3.)? * self.chi.ln() + x.ln() + S::HALF * (-x * x).ln_1p()
                - self.a * (S::ONE - x * x)
                - S::HALF * crate::distribution::special::log_two_pi::<S>()
                + S::LN_2
                - self.log_norm,
        )
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if x >= S::ONE {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        if !upper && self.a * x * x > S::ONE {
            // Once the gamma interval is well separated, direct upper gamma
            // tails retain lower ARGUS mass even when both lower gamma tails
            // round to one (for example chi=100 and x=0.9).
            let shape = number::<S>(1.5)?;
            return Ok(crate::law::logdiff(
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_log_q_with_options(
                        shape,
                        self.a * (S::ONE - x * x),
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?,
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_log_q_with_options(
                        shape,
                        self.a,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?,
                scioxide_core::argument!("self"),
            )? - self.log_norm);
        }
        if !upper && x < S::HALF {
            // u=(t/x)^2 removes the t factor in the density. The remaining
            // positive integral is scaled to [0,1], retaining 2*ln(x) even
            // when x*x underflows. No near-one gamma ratio is subtracted.
            let log_coefficient = if self.a == S::ZERO {
                number::<S>(1.5)?.ln()
            } else {
                number::<S>(3.)? * self.chi.ln()
                    - S::HALF * crate::distribution::special::log_two_pi::<S>()
                    - self.log_norm
                    - self.a
            };
            let mass = crate::distribution::quadrature::positive_integral(
                S::ZERO,
                S::ONE,
                self.options.tolerance().relative(),
                left,
                &|u| {
                    let z = x * x * u;
                    Ok(S::HALF * (-z).ln_1p() + self.a * z)
                },
                None,
            )?;
            return Ok(log_coefficient + S::TWO * x.ln() + mass);
        }
        let log_upper = if self.a == S::ZERO {
            number::<S>(1.5)? * (-x * x).ln_1p()
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::gamma_log_p_with_options(
                    number::<S>(1.5)?,
                    self.a * (S::ONE - x * x),
                    scioxide_special::GammaOptions::new(budget),
                )
            })? - self.log_norm
        };
        if upper {
            Ok(log_upper)
        } else {
            log1mexp(log_upper)
        }
    }

    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if upper {
            if p == S::ONE {
                return Ok(S::ZERO);
            }
            if p == S::ZERO {
                return Ok(S::ONE);
            }
            if p > S::HALF {
                return self.inverse_with_remaining(S::ONE - p, false, left);
            }
            if self.a == S::ZERO {
                return finite((-((number::<S>(2.)? / number::<S>(3.)?) * p.ln()).exp_m1()).sqrt());
            }
            let y = crate::law::special(left, |budget| {
                scioxide_special::gamma_p_inv_with_options(
                    number::<S>(1.5)?,
                    p * self.log_norm.exp(),
                    scioxide_special::GammaOptions::new(budget),
                )
            })?;
            finite((S::ONE - y / self.a).sqrt())
        } else {
            if p == S::ZERO {
                return Ok(S::ZERO);
            }
            if p == S::ONE {
                return Ok(S::ONE);
            }
            if p > S::HALF {
                return self.inverse_with_remaining(S::ONE - p, true, left);
            }
            // Invert in ln(x), avoiding an absolute coordinate tolerance near zero.
            let log_p = p.into_f64().ln();
            let options = crate::inversion::smooth_options(self.work_budget(), log_p)?;
            let guess = 0.5 * log_p;
            let log_x = crate::inversion::invert_smooth_remaining(
                log_p,
                false,
                guess,
                (f64::NEG_INFINITY, 0.0),
                &options,
                |y, upper, left| {
                    let x = S::try_from_f64_lossy(y.exp())?;
                    Ok(self.log_probability_with_work(x, upper, left)?.into_f64())
                },
                |y, _| {
                    let x = S::try_from_f64_lossy(y.exp())?;
                    Ok(self.density_log(x)?.into_f64() + y)
                },
                left,
            )?;
            S::try_from_f64_lossy(log_x.exp())
        }
    }
}
formula_law!(Argus);
impl<S: RealScalar> Mean<S> for Argus<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.mean_work(&mut self.options.work_budget().get())
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.mean_work(left)
        })
    }
}
impl<S: RealScalar> Argus<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        if self.a == S::ZERO {
            return finite(number::<S>(3.)? * S::PI / number::<S>(16.)?);
        }
        let h = crate::law::special(left, |budget| {
            scioxide_special::hyp1f1_with_options(
                number::<S>(1.5)?,
                number::<S>(3.)?,
                -self.a,
                scioxide_special::HypergeometricOptions::default().with_iteration_budget(budget),
            )
        })?;
        crate::law::positive_result(h)?;
        finite(
            (scioxide_special::ln_beta(number::<S>(1.5)?, number::<S>(1.5)?)?
                - scioxide_special::ln_gamma(number::<S>(1.5)?)?
                + number::<S>(1.5)? * self.a.ln()
                + h.ln()
                - self.log_norm)
                .exp(),
        )
    }
}

impl<S: RealScalar> Variance<S> for Argus<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        self.variance_work(&mut self.options.work_budget().get())
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.variance_work(left)
        })
    }
}
impl<S: RealScalar> Argus<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        variance_from_moments(self.second_raw(left)?, self.mean_work(left)?)
    }
}
