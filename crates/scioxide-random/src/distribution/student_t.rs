//! Student's t-distribution.

use scioxide_core::backend::checked_final_rounding;
use scioxide_core::checked_float_product;
use scioxide_core::{RealScalar, SciOxideError};

use crate::distribution::chi_squared::ChiSquareKernel;
use crate::distribution::normal::Normal;
use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

/// Student's t-distribution with `ν` degrees of freedom.
///
/// ```text
/// f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2),   x ∈ ℝ, ν > 0.
/// ```
///
/// The law is standard. Shift and scale it with [`Affine`](crate::Affine). A
/// sample is `Z / √(V/ν)` for independent `Z ~ Normal(0, 1)` and
/// `V ~ χ²(ν)`. The quantiles at `p = 0` and `p = 1` are `−∞` and `+∞`. Every
/// other quantile is finite, and the median is exactly 0.
///
/// Tails and quantiles use the incomplete beta function in whichever of
/// `t = ν/(ν + x²)` and `u = x²/(ν + x²)` is at most 1/2, so neither is
/// computed as one minus the other. For `ν ≥ 10¹⁰` the tails use Hill's
/// normalizing transformation. For `ν ≥ 10⁶` the quantiles use the
/// Cornish–Fisher expansion through `ν⁻⁴`. In both ranges the truncation error
/// is below the rounding error of `f64`. For `f32` the quantile is computed in
/// `f64` and rounded once.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, InverseSurvival, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, StudentT, Variance};
/// let law = StudentT::new(5.0_f64)?;
/// assert_eq!(law.mean()?, 0.0);
/// assert!((law.variance()? - 5.0 / 3.0).abs() < 1e-15); // ν/(ν-2)
/// assert_eq!(law.quantile(0.5)?, 0.0);
/// assert!(law.quantile(0.5 + 1e-12)? > 0.0);
/// assert_eq!(law.isf(0.5 + 1e-12)?, -law.quantile(0.5 + 1e-12)?);
/// let wide = StudentT::new(1e14_f64)?;
/// assert!((wide.cdf(-0.1)? - 0.460_172_162_722_971).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(22);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. W. Hill, "Algorithm 395: Student's t-distribution", *Commun. ACM*
/// 13(10), 1970, 617–619.
#[derive(Debug, Clone)]
#[must_use]
pub struct StudentT<S: RealScalar> {
    options: crate::LawOptions<S>,
    nu: S,
    root_nu: S,
    log_root_nu: S,
    density_power: S,
    log_beta: Result<S, SciOxideError>,
    normal: Normal<S>,
    chi2: ChiSquareKernel<S>,
    // Two-sided tail at |x| = sqrt(ν), where t = u = 1/2. Unused from
    // `QUANTILE_EXPANSION_DEGREES`.
    split: f64,
}

// Cornish–Fisher terms through ν⁻⁴ (Abramowitz–Stegun 26.7.5). For every
// binary64 probability |z| < 39, so the first omitted term is below rounding.
const QUANTILE_EXPANSION_DEGREES: f64 = 1e6;
// Hill (1970) tail transform; its truncation error falls as ν⁻² and is below
// binary64 rounding here. The incomplete-beta kernels cover smaller degrees.
const TAIL_EXPANSION_DEGREES: f64 = 1e10;

impl<S: RealScalar> StudentT<S> {
    /// Creates a Student's t-distribution with `nu` degrees of freedom.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `nu` is NaN or infinite.
    /// - `DomainError` if `nu ≤ 0`.
    pub fn new(nu: S) -> Result<Self, SciOxideError> {
        Self::with_options(nu, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(nu: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::positive(nu, scioxide_core::argument!("nu"))?;
        let normal = Normal::standard();
        let chi2 = ChiSquareKernel::new(nu)?;
        let degrees = nu.into_f64();
        let split = if degrees < QUANTILE_EXPANSION_DEGREES {
            crate::law::special(&mut options.work_budget().get(), |budget| {
                scioxide_special::beta_inc_with_options(
                    0.5 * degrees,
                    0.5,
                    0.5,
                    scioxide_special::BetaOptions::new(budget),
                )
            })?
        } else {
            0.0
        };
        Ok(Self {
            options,
            nu,
            root_nu: nu.sqrt(),
            log_root_nu: nu.ln() * S::HALF,
            density_power: (nu + S::ONE) * S::HALF,
            log_beta: scioxide_special::ln_beta(nu * S::HALF, S::HALF),
            normal,
            chi2,
            split,
        })
    }

    /// Two-sided tail `P(|T| > |x|)`, or its logarithm.
    fn two_sided(&self, x: S, log: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let half = S::HALF;
        let magnitude = x.abs();
        if magnitude == S::ZERO {
            return Ok(if log { S::ZERO } else { S::ONE });
        }
        let a = self.nu * half;
        let r = magnitude / self.root_nu;
        let expansion = self.nu >= S::try_from_f64_lossy(TAIL_EXPANSION_DEGREES)?;
        if r <= S::ONE {
            if expansion {
                return self.hill_tail(magnitude, r, log);
            }
            // u = r²/(1+r²) <= 1/2; the tail is the upper incomplete beta in u.
            let square = r * r;
            let u = square / (S::ONE + square);
            return if log {
                crate::law::special(left, |budget| {
                    scioxide_special::beta_inc_log_q_with_options(
                        half,
                        a,
                        u,
                        scioxide_special::BetaOptions::new(budget),
                    )
                })
            } else {
                crate::law::special(left, |budget| {
                    scioxide_special::beta_inc_q_with_options(
                        half,
                        a,
                        u,
                        scioxide_special::BetaOptions::new(budget),
                    )
                })
            };
        }
        // t = 1/(1+r²) < 1/2, formed from 1/r so neither r² nor t overflows.
        let inverse = S::ONE / r;
        let square = inverse * inverse;
        let log_t = S::TWO * inverse.ln() - square.ln_1p();
        if let Some(value) = crate::distribution::special::beta_leading_log(a, half, log_t, || {
            self.log_beta.clone()
        })? {
            return Ok(if log { value } else { value.exp() });
        }
        let t = square / (S::ONE + square);
        if expansion {
            // I_t(a, 1/2) = t^a / (a B(a, 1/2)) * sum_k (1/2)_k / k! * a / (a + k) * t^k,
            // which converges at least like 2^-k for t <= 1/2 and any a.
            let mut power = S::ONE;
            let mut sum = S::ONE;
            let mut k = S::ONE;
            loop {
                crate::law::charge(left)?;
                power = power * (k - half) / k * t;
                let term = power * a / (a + k);
                sum += term;
                if term <= S::EPSILON * sum || k >= S::try_from_f64_lossy(128.)? {
                    break;
                }
                k += S::ONE;
            }
            let value = a * log_t - a.ln() - self.log_beta.clone()? + sum.ln();
            return Ok(if log { value } else { value.exp() });
        }
        if log {
            crate::law::special(left, |budget| {
                scioxide_special::beta_inc_log_with_options(
                    a,
                    half,
                    t,
                    scioxide_special::BetaOptions::new(budget),
                )
            })
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::beta_inc_with_options(
                    a,
                    half,
                    t,
                    scioxide_special::BetaOptions::new(budget),
                )
            })
        }
    }

    // Hill (1970, Algorithm 395): w(z) with z² = (ν - 1/2) ln(1 + x²/ν) is a
    // normal deviate. For r = |x|/sqrt(ν) <= 1 and ν >= 1e10 its error is
    // below 1e-17 relative in the log tail. Terms use l = z²/(ν - 1/2), so no
    // power of z or ν overflows, and z is formed from |x| directly.
    fn hill_tail(&self, magnitude: S, r: S, log: bool) -> Result<S, SciOxideError> {
        let number = S::try_from_f64_lossy;
        let c = self.nu - S::HALF;
        let v = S::ONE / c;
        let square = r * r;
        let ratio = if square == S::ZERO {
            S::ONE
        } else {
            square.ln_1p() / square
        };
        let z = magnitude * ((S::ONE - S::HALF / self.nu) * ratio).sqrt();
        let l = z * z * v;
        let first = (l + number(3.)? * v) * v / number(48.)?;
        let second = (((number(0.4)? * l + number(3.3)? * v) * l + number(24.)? * v * v) * l
            + number(85.5)? * v * v * v)
            * v
            / (number(48.)? * (number(48.)? + number(0.8)? * l * l + number(100.)? * v * v));
        let w = z * (S::ONE + first - second);
        if log {
            Ok(scioxide_special::normal_log_sf(w)? + S::LN_2)
        } else {
            checked_final_rounding(
                S::TWO * scioxide_special::normal_sf(w)?,
                "Student-t tail exceeds the finite scalar range",
            )
        }
    }

    /// `f64` quantile for an interior probability.
    fn quantile_f64(&self, p: f64, left: &mut usize) -> Result<f64, SciOxideError> {
        if p == 0.5 {
            return Ok(0.0);
        }
        let nu = self.nu.into_f64();
        if nu >= QUANTILE_EXPANSION_DEGREES {
            let z = scioxide_special::normal_quantile(p)?;
            let s = z * z;
            let v = 1.0 / nu;
            let g1 = (s + 1.0) / 4.0;
            let g2 = ((5.0 * s + 16.0) * s + 3.0) / 96.0;
            let g3 = (((3.0 * s + 19.0) * s + 17.0) * s - 15.0) / 384.0;
            let g4 = ((((79.0 * s + 776.0) * s + 1482.0) * s - 1920.0) * s - 945.0) / 92160.0;
            return Ok(z + z * v * (g1 + v * (g2 + v * (g3 + v * g4))));
        }
        let a = 0.5 * nu;
        // Both 1 - p for p > 1/2 and the doubling are exact.
        let (q, sign) = if p > 0.5 {
            (2.0 * (1.0 - p), 1.0)
        } else {
            (2.0 * p, -1.0)
        };
        let magnitude = if q <= self.split {
            // t = ν/(ν+x²) <= 1/2.
            let t = crate::law::special(left, |budget| {
                scioxide_special::beta_inc_inv_with_options(
                    a,
                    0.5,
                    q,
                    scioxide_special::BetaOptions::new(budget),
                )
            })?;
            if t.is_nan() || t > 0.5 {
                return Err(unresolved());
            }
            if t < f64::MIN_POSITIVE {
                leading_magnitude(nu, a, q)?
            } else {
                checked_float_product(nu.sqrt(), (1.0 - t).sqrt() / t.sqrt())?
            }
        } else {
            // u = x²/(ν+x²) < 1/2. Near the median its probability
            // 2|p - 1/2| is exact; otherwise invert the upper tail q.
            let central = 2.0 * (p - 0.5).abs();
            let u = if central <= 0.5 {
                crate::law::special(left, |budget| {
                    scioxide_special::beta_inc_inv_with_options(
                        0.5,
                        a,
                        central,
                        scioxide_special::BetaOptions::new(budget),
                    )
                })?
            } else {
                crate::law::special(left, |budget| {
                    scioxide_special::beta_inc_q_inv_with_options(
                        0.5,
                        a,
                        q,
                        scioxide_special::BetaOptions::new(budget),
                    )
                })?
            };
            if u.is_nan() || u > 0.5 {
                return Err(unresolved());
            }
            checked_float_product(nu.sqrt(), u.sqrt() / (1.0 - u).sqrt())?
        };
        if magnitude == 0.0 {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "non-median Student-t quantile rounded to zero",
            ));
        }
        checked_float_product(sign, magnitude)
    }

    fn quantile_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        // Binary32 coordinates can underflow while the quantile is finite.
        S::try_from_f64_lossy(self.quantile_f64(p.into_f64(), left)?)
    }

    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let log = self.two_sided(x, true, left)? - S::LN_2;
        Ok(if x >= S::ZERO {
            log
        } else {
            crate::law::log1mexp(log)?
        })
    }

    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_sf_work(-x, left)
    }

    fn isf_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        // Symmetry: the upper-tail quantile at p is minus the lower one.
        self.quantile_work(p, left).map(|x| -x)
    }
}

fn unresolved() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "Student-t beta coordinate left its half interval",
    )
}

// Below the binary64 normal range I_t(a, 1/2) = t^a / (a B(a, 1/2)) exactly to
// working precision, and x = sqrt(ν(1-t)/t) = sqrt(ν) (q a B)^(-1/ν). Here
// a B(a, 1/2) >= 1, so only a subnormal q needs scaling; the power of two is
// exact for ν = 1, 2, 4, ...
fn leading_magnitude(nu: f64, a: f64, q: f64) -> Result<f64, SciOxideError> {
    let (q, rescale) = if q < f64::MIN_POSITIVE {
        // 2^128.
        (
            q * f64::from_bits(0x47f0_0000_0000_0000),
            2_f64.powf(128.0 / nu),
        )
    } else {
        (q, 1.0)
    };
    let base = q * (a.ln() + scioxide_special::ln_beta(a, 0.5)?).exp();
    let power = base.powf(-1.0 / nu) * rescale;
    checked_float_product(
        nu.sqrt(),
        checked_final_rounding(
            power,
            "interior Student-t quantile is outside the finite scalar range",
        )?,
    )
}

impl<S: RealScalar> Distribution<S> for StudentT<S> {
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let z: S = self.normal.sample(rng)?;
        // Definition T = Z / sqrt(V / nu) with independent Z ~ N(0, 1) and
        // V ~ chi^2(nu). A zero denominator can only arise from underflow for
        // tiny nu and is surfaced instead of producing an infinite sample.
        let chi2_val: S = self.chi2.sample(rng)?;
        let denominator = (chi2_val / self.nu).sqrt();
        if !denominator.is_finite() || denominator <= S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "chi-squared normalization did not produce a positive finite denominator",
            ));
        }
        checked_final_rounding(
            z / denominator,
            "Student-t sample exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Density<S> for StudentT<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        let log_pdf = crate::LogDensity::log_pdf(self, x)?;
        checked_final_rounding(
            log_pdf.exp(),
            "Student-t density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for StudentT<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::Survival::sf(self, -x)
    }
}

impl<S: RealScalar> Quantile<S> for StudentT<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.quantile_work(p, &mut self.options.work_budget().get())
    }

    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.quantile_work(p, left)
        })
    }
}

impl<S: RealScalar> Mean<S> for StudentT<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.nu > S::ONE {
            Ok(S::ZERO)
        } else {
            Err(crate::law::undefined_moment(
                "mean is undefined for nu <= 1",
            ))
        }
    }
}

impl<S: RealScalar> Variance<S> for StudentT<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let two = S::TWO;
        if self.nu > two {
            checked_final_rounding(
                self.nu / (self.nu - two),
                "Student-t variance exceeds the finite scalar range",
            )
        } else {
            Err(crate::law::undefined_moment(
                "variance is undefined for nu <= 2",
            ))
        }
    }
}

impl<S: RealScalar> crate::Support<S> for StudentT<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(None, None)
    }
}

impl<S: RealScalar> crate::LogDensity<S> for StudentT<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        // ln(1 + r^2), split so that r^2 cannot overflow for large r:
        // ln(1 + r^2) = 2 ln r + ln(1 + r^-2).
        let r = x.abs() / self.root_nu;
        let log = if r > S::ONE {
            let log_r = if r.is_finite() {
                r.ln()
            } else {
                x.abs().ln() - self.log_root_nu
            };
            S::TWO * log_r + (S::ONE / r / r).ln_1p()
        } else {
            (r * r).ln_1p()
        };
        Ok(-self.log_beta.clone()? - self.log_root_nu - self.density_power * log)
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for StudentT<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_sf_work(x, &mut self.options.work_budget().get())
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_sf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::LogCdf<S> for StudentT<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_cdf_work(x, &mut self.options.work_budget().get())
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_cdf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::Survival<S> for StudentT<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let half_tail = self.two_sided(x, false, &mut self.options.work_budget().get())? * S::HALF;
        Ok(if x >= S::ZERO {
            half_tail
        } else {
            S::ONE - half_tail
        })
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for StudentT<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.isf_work(p, &mut self.options.work_budget().get())
    }

    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.isf_work(p, left)
        })
    }
}
impl<S: RealScalar> crate::Entropy<S> for StudentT<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(super::moment_math::student_entropy(self.nu.into_f64())?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::SeedableRng;

    #[test]
    fn test_new_valid() {
        assert!(StudentT::<f64>::new(5.0).is_ok());
    }

    #[test]
    fn test_new_invalid() {
        let error = StudentT::<f64>::new(0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite() {
        let error = StudentT::<f64>::new(f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_finite() {
        let dist = StudentT::new(3.0_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x.is_finite());
        }
    }

    #[test]
    fn test_sample_composes_default_normal_then_chi_squared_streams() {
        let nu = 7.0_f64;
        let dist = StudentT::new(nu).unwrap();
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let chi_squared = crate::ChiSquared::new(nu).unwrap();
        let mut expected_rng = SplitMix64::from_seed(83);
        let numerator = normal.sample(&mut expected_rng).unwrap();
        let denominator = (chi_squared.sample(&mut expected_rng).unwrap() / nu).sqrt();

        let mut actual_rng = SplitMix64::from_seed(83);
        assert_eq!(
            dist.sample(&mut actual_rng).unwrap(),
            numerator / denominator
        );
    }

    #[test]
    fn test_pdf_symmetric() {
        let dist = StudentT::new(5.0_f64).unwrap();
        let p1 = dist.pdf(1.0).unwrap();
        let p2 = dist.pdf(-1.0).unwrap();
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_at_zero() {
        let dist = StudentT::new(10.0_f64).unwrap();
        assert!((dist.cdf(0.0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cdf_symmetry() {
        let dist = StudentT::new(5.0_f64).unwrap();
        let c1 = dist.cdf(1.5).unwrap();
        let c2 = dist.cdf(-1.5).unwrap();
        assert!((c1 + c2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_median() {
        let dist = StudentT::new(10.0_f64).unwrap();
        assert!((dist.quantile(0.5).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn quantile_endpoints_are_exact_support_limits() {
        let dist = StudentT::new(10.0_f64).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = StudentT::new(5.0_f64).unwrap();
        let p = 0.9;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-6, "p_back={p_back}");
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = StudentT::new(5.0_f64).unwrap();
        let error = dist.quantile(f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_mean_defined() {
        let dist = StudentT::new(5.0_f64).unwrap();
        assert!((dist.mean().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_mean_undefined() {
        let dist = StudentT::new(1.0_f64).unwrap();
        let error = dist.mean().unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_variance() {
        let dist = StudentT::new(5.0_f64).unwrap();
        // Var = 5/(5-2) = 5/3
        assert!((dist.variance().unwrap() - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_variance_undefined() {
        let dist = StudentT::new(2.0_f64).unwrap();
        let error = dist.variance().unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }
}
