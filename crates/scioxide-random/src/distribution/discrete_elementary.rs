//! Elementary exponential lattice laws and signed integer uniform sampling.
use super::count_math::analytic_count;
use super::count_math::{self as math};
use super::count_math::{Kernel, count_capabilities, signed_capabilities};
use crate::distribution::LawOptions;
use crate::law::{log1mexp, logadd};
use crate::{BitGenerator, CountSupport, SignedCountSupport};
use scioxide_core::{RealScalar, SciOxideError};

/// Boltzmann distribution: a discrete exponential law truncated to `0, …, n − 1`.
///
/// With rate `λ > 0` and `n ≥ 1`, the mass function is
///
/// ```text
/// P(k) = (1 − e^(−λ)) e^(−λ k) / (1 − e^(−λ n)),   k = 0, …, n − 1.
/// ```
///
/// ```
/// use scioxide_random::{Boltzmann, Cdf, CountDistribution, Mass, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// // lambda = ln 2, n = 3: masses 4/7, 2/7, 1/7.
/// let law = Boltzmann::new(core::f64::consts::LN_2, 3)?;
/// assert!((law.pmf(1)? - 2.0 / 7.0).abs() < 1e-15);
/// assert!((law.cdf(0)? - 4.0 / 7.0).abs() < 1e-15 && (law.sf(0)? - 3.0 / 7.0).abs() < 1e-15);
/// assert_eq!(law.quantile(0.5)?, 0);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? < 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Boltzmann<S: RealScalar> {
    rate: f64,
    n: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> Boltzmann<S> {
    /// Creates a Boltzmann law with rate `lambda` on `0, …, n − 1`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lambda` is NaN or infinite.
    /// - `DomainError` if `lambda ≤ 0`.
    /// - `InvalidInput` if `n = 0`.
    /// - `NotImplemented` if `n > 2⁵³`.
    pub fn new(lambda: S, n: usize) -> Result<Self, SciOxideError> {
        Self::with_options(lambda, n, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        lambda: S,
        n: usize,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let rate = math::parameter(lambda, true, scioxide_core::argument!("lambda"))?;
        if n == 0 {
            return Err(crate::law::invalid(
                scioxide_core::argument!("n"),
                "Boltzmann support must be nonempty",
            ));
        }
        analytic_count(n)?;
        Ok(Self { rate, n, options })
    }
    fn normalizer(&self) -> Result<f64, SciOxideError> {
        log1mexp(-self.rate * self.n as f64)
    }
    fn moments(&self) -> (f64, f64) {
        let n = self.n as f64;
        if self.n == 1 {
            return (0.0, 0.0);
        }
        let x = self.rate * n;
        if x < 0.1 {
            // Bernoulli-number expansions of 1/expm1(z) and its derivative.
            // At |z|<0.1 the first omitted variance term is below 6e-20*n².
            let h = |z: f64| {
                z * (1.0 / 12.0
                    + z * z * (-1.0 / 720.0 + z * z * (1.0 / 30240.0 - z * z / 1209600.0)))
            };
            let v = |z: f64| {
                z * z
                    * (1.0 / 240.0
                        + z * z * (-1.0 / 6048.0 + z * z * (1.0 / 172800.0 - z * z / 5322240.0)))
            };
            (
                (n - 1.0) * 0.5 + h(self.rate) - n * h(x),
                (n * n - 1.0) / 12.0 + v(self.rate) - n * n * v(x),
            )
        } else {
            let q = (-self.rate).exp();
            let p = -(-self.rate).exp_m1();
            let qn = (-x).exp();
            let pn = -(-x).exp_m1();
            (q / p - n * qn / pn, q / p / p - n * n * qn / pn / pn)
        }
    }
}
impl<S: RealScalar> CountSupport for Boltzmann<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (0, Some(self.n - 1))
    }
}
impl<S: RealScalar> Kernel<S> for Boltzmann<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k >= self.n {
            return Ok(f64::NEG_INFINITY);
        }
        if self.n == 1 {
            return Ok(0.0);
        }
        Ok(log1mexp(-self.rate)? - self.normalizer()? - crate::law::finite(self.rate * k as f64)?)
    }
    fn tail_with_work(
        &self,
        k: usize,
        upper: bool,
        _left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        if k >= self.n - 1 {
            return Ok(if upper { f64::NEG_INFINITY } else { 0.0 });
        }
        // Truncated geometric: F(k) = (1 - e^{-r(k+1)}) / (1 - e^{-rn}) and
        // S(k) = e^{-r(k+1)} (1 - e^{-r(n-k-1)}) / (1 - e^{-rn}). Both are closed
        // forms in log space, so neither tail is formed by complementing.
        let x = self.rate * (k + 1) as f64;
        Ok(if upper {
            -crate::law::finite(x)? + log1mexp(-self.rate * (self.n - k - 1) as f64)?
                - self.normalizer()?
        } else {
            log1mexp(-x)? - self.normalizer()?
        })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(self.moments().0)
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        Ok(self.moments().1)
    }
}
count_capabilities!(Boltzmann);

/// Discrete Laplace distribution on the integers with rate `a > 0`.
///
/// The mass function is `P(k) = tanh(a/2) e^(−a |k|)` for `k ∈ ℤ`. Counts are
/// `i64`. Draws invert the distribution function.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, DiscreteLaplace, Mass, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// // a = ln 2: tanh(a/2) = 1/3, so P(k) = 2^-|k| / 3.
/// let law = DiscreteLaplace::new(core::f64::consts::LN_2)?;
/// assert!((law.pmf(0)? - 1.0 / 3.0).abs() < 1e-15 && (law.pmf(-1)? - 1.0 / 6.0).abs() < 1e-15);
/// assert!((law.cdf(-1)? - 1.0 / 3.0).abs() < 1e-15 && (law.sf(0)? - 1.0 / 3.0).abs() < 1e-15);
/// assert_eq!(law.quantile(0.5)?, 0);
/// let k: i64 = law.sample(&mut SplitMix64::from_seed(42))?;
/// assert!(law.pmf(k)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DiscreteLaplace<S: RealScalar> {
    rate: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> DiscreteLaplace<S> {
    fn tail_with_work(&self, k: i64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        crate::law::charge(left)?;
        self.tail(k, upper)
    }
    /// Creates a discrete Laplace law with rate `a`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0`.
    pub fn new(a: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(a: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            rate: math::parameter(a, true, scioxide_core::argument!("a"))?,
            options,
        })
    }
    fn mass(&self, k: i64) -> Result<f64, SciOxideError> {
        let x = super::count_math::signed_count_value(k)?;
        Ok(log1mexp(-self.rate)?
            - logadd(0.0, -self.rate)?
            - crate::law::finite(self.rate * x.abs())?)
    }
    fn tail(&self, k: i64, upper: bool) -> Result<f64, SciOxideError> {
        let x = super::count_math::signed_count_value(k)?;
        // The tail on the far side of zero is a geometric sum,
        // e^{-a m} / (1 + e^{-a}) with m the distance to the tail start; the
        // near-side tail is its complement.
        let small = -self.rate * (if k < 0 { -x } else { x + 1.0 }) - logadd(0.0, -self.rate)?;
        Ok(if upper == (k >= 0) {
            crate::law::finite(small)?
        } else {
            log1mexp(small)?
        })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        let p = -(-self.rate).exp_m1();
        Ok(2.0 * (-self.rate).exp() / p / p)
    }
    fn inverse(&self, p: S, upper: bool, left: &mut usize) -> Result<i64, SciOxideError> {
        math::inverse_signed(p, upper, None, None, left, |k, u, _| self.tail(k, u))
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<i64, SciOxideError> {
        math::inverse_signed(
            math::open(rng)?,
            false,
            None,
            None,
            &mut self.options.work_budget().get(),
            |k, u, _| self.tail(k, u),
        )
    }
}
impl<S: RealScalar> SignedCountSupport for DiscreteLaplace<S> {
    fn signed_count_support(&self) -> (Option<i64>, Option<i64>) {
        (None, None)
    }
}

/// Discrete uniform distribution on the integers `low, …, high − 1`.
///
/// Each integer has mass `1 / (high − low)`. Draws use Lemire's nearly
/// divisionless rejection on uniform integer words, so every `i64` range is
/// sampled without bias.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, DiscreteUniform, Mass, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = DiscreteUniform::<f64>::new(-2, 3)?; // counts -2..=2
/// assert!((law.pmf(0)? - 0.2).abs() < 1e-15);
/// assert!((law.cdf(0)? - 0.6).abs() < 1e-15 && (law.sf(0)? - 0.4).abs() < 1e-15);
/// assert_eq!(law.quantile(0.5)?, 0);
/// assert!((-2..3).contains(&law.sample(&mut SplitMix64::from_seed(42))?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. Lemire, "Fast random integer generation in an interval", *ACM Trans.
/// Model. Comput. Simul.* 29(1), 2019, article 3.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteUniform<S: RealScalar> {
    low: i64,
    high: i64,
    options: LawOptions<S>,
}
impl<S: RealScalar> DiscreteUniform<S> {
    fn tail_with_work(&self, k: i64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        crate::law::charge(left)?;
        self.tail(k, upper)
    }
    /// Creates a discrete uniform law on the half-open range `[low, high)`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `low ≥ high`.
    pub fn new(low: i64, high: i64) -> Result<Self, SciOxideError> {
        Self::with_options(low, high, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        low: i64,
        high: i64,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if low >= high {
            return Err(crate::law::domain(
                scioxide_core::argument!("high"),
                "integer uniform requires low < high",
            ));
        }
        Ok(Self { low, high, options })
    }
    fn width(&self) -> u64 {
        (i128::from(self.high) - i128::from(self.low)) as u64
    }
    // These closed-form uniform kernels cannot fail.
    fn mass(&self, k: i64) -> Result<f64, SciOxideError> {
        Ok(if k < self.low || k >= self.high {
            f64::NEG_INFINITY
        } else {
            -(self.width() as f64).ln()
        })
    }
    fn tail(&self, k: i64, upper: bool) -> Result<f64, SciOxideError> {
        if k < self.low {
            return Ok(if upper { 0.0 } else { f64::NEG_INFINITY });
        }
        if k >= self.high - 1 {
            return Ok(if upper { f64::NEG_INFINITY } else { 0.0 });
        }
        let lower = (i128::from(k) - i128::from(self.low) + 1) as u64;
        let higher = self.width() - lower;
        let small = (lower.min(higher) as f64 / self.width() as f64).ln();
        Ok(if upper == (higher < lower) {
            small
        } else {
            log1mexp(small)?
        })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok((i128::from(self.low) + i128::from(self.high) - 1) as f64 * 0.5)
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        let n = self.width() as f64;
        Ok((n - 1.0) * (n + 1.0) / 12.0)
    }
    fn inverse(&self, p: S, upper: bool, left: &mut usize) -> Result<i64, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        crate::law::charge(left)?;
        let p = p.into_f64();
        if p == 0.0 {
            return Ok(if upper { self.high - 1 } else { self.low });
        }
        if p == 1.0 {
            return Ok(if upper { self.low } else { self.high - 1 });
        }
        // Exact dyadic p times an exact u64 width, avoiding count rounding above 2^53.
        let bits = p.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u32;
        let significand =
            (bits & ((1_u64 << 52) - 1)) | if exponent == 0 { 0 } else { 1_u64 << 52 };
        let shift = if exponent == 0 { 1074 } else { 1075 - exponent };
        let product = u128::from(significand) * u128::from(self.width());
        let (floor, remainder) = if shift >= 128 {
            (0, product != 0)
        } else {
            (product >> shift, product & ((1_u128 << shift) - 1) != 0)
        };
        Ok(if upper {
            (i128::from(self.high) - 1 - floor as i128) as i64
        } else {
            (i128::from(self.low) + floor as i128 + i128::from(remainder) - 1) as i64
        })
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<i64, SciOxideError> {
        if self.width() == 1 {
            return Ok(self.low);
        }
        Ok(
            (i128::from(self.low) + i128::from(crate::sampling::uniform_below(self.width(), rng)?))
                as i64,
        )
    }
}
impl<S: RealScalar> SignedCountSupport for DiscreteUniform<S> {
    fn signed_count_support(&self) -> (Option<i64>, Option<i64>) {
        (Some(self.low), Some(self.high - 1))
    }
}

signed_capabilities!(DiscreteLaplace, sampled);
signed_capabilities!(DiscreteUniform, sampled);
