//! Generalized inverse Gaussian distribution and the normal variance-mean mixtures built on it.
use crate::distribution::LawOptions;
use crate::distribution::formula::{Formula, Inverse, formula_law};
use crate::distribution::gamma::GammaKernel;
use crate::distribution::quadrature::tail;
use crate::distribution::special::{expm1_minus, log_k};
use crate::law::{argument, constant as c, finite, log1mexp};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Generalized inverse Gaussian (GIG) distribution on `x > 0` with index `p` and
/// concentration `b > 0`.
///
/// ```text
/// f(x) = x^(p−1) exp(−b (x + 1/x) / 2) / (2 K_p(b)),
/// ```
///
/// where `K_p` is the modified Bessel function of the second kind (DLMF 10.25).
/// The law has unit scale; scale it with [`Affine`](crate::Affine). Tail
/// probabilities are computed by quadrature in `ln x`. Draws use rejection from
/// a three-piece envelope in `ln x`, with three open uniforms per proposal.
///
/// # References
///
/// B. Jørgensen, *Statistical Properties of the Generalized Inverse Gaussian
/// Distribution*, Lecture Notes in Statistics 9, Springer, 1982.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, GeneralizedInverseGaussian, Mean, Quantile, SeedableRng,
///     SplitMix64,
/// };
/// let law = GeneralizedInverseGaussian::<f64>::new(1.0, 2.0)?;
/// let x = law.quantile(0.6)?;
/// assert!((law.cdf(x)? - 0.6).abs() < 1e-10);
/// assert!(law.mean()? > 0.0);
/// let mut rng = SplitMix64::from_seed(51);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedInverseGaussian<S: RealScalar> {
    p: S,
    b: S,
    log_norm: S,
    mode: S,
    width: S,
    /// Rejection envelope prepared once; its failure fails only sampling.
    envelope: Result<GigEnvelope<S>, SciOxideError>,
    options: LawOptions<S>,
}

/// Three-piece rejection envelope of a GIG in `t = ln x - mode` (for `|p|`).
/// In `t` the log density is `-loss(t)` with `loss(t) = a (e^t - 1 - t) + b'
/// (e^-t - 1 + t)`, convex with minimum 0 at `t = 0`. The envelope is flat
/// (value 1) on `[left, right]`, whose ends have `loss >= 1`, and uses
/// tangent-line exponential bounds outside: convexity and `loss >= 0` make
/// `loss(t) >= loss'(edge) (t - edge)`, so it dominates the target without
/// further constants.
#[derive(Debug, Clone, Copy)]
struct GigEnvelope<S> {
    mode: S,
    a: S,
    b: S,
    left: S,
    right: S,
    sl: S,
    sr: S,
    al: S,
    ar: S,
    area: S,
}

impl<S: RealScalar> GigEnvelope<S> {
    fn new(p: S, b_parameter: S) -> Result<Self, SciOxideError> {
        let p = p.abs();
        let r = p.hypot(b_parameter);
        let mode = if p == S::ZERO {
            S::ZERO
        } else {
            (p + r).ln() - b_parameter.ln()
        };
        let a = (r + p) * S::HALF;
        let b = b_parameter * S::HALF * (b_parameter / (r + p));
        let mut envelope = Self {
            mode,
            a,
            b,
            left: -(S::ONE / r.sqrt()).min(S::ONE),
            right: S::ZERO,
            sl: S::ZERO,
            sr: S::ZERO,
            al: S::ZERO,
            ar: S::ZERO,
            area: S::ZERO,
        };
        envelope.right = -envelope.left;
        // Fixed bounded preparation of the three-piece envelope.
        for _ in 0..64 {
            if envelope.loss(envelope.left)? >= S::ONE {
                break;
            }
            envelope.left *= c(2.)?;
        }
        for _ in 0..64 {
            if envelope.loss(envelope.right)? >= S::ONE {
                break;
            }
            envelope.right *= c(2.)?;
        }
        let (left, right) = (envelope.left, envelope.right);
        envelope.sl = -a * left.exp_m1() + b * (-left).exp_m1();
        envelope.sr = a * right.exp_m1() - b * (-right).exp_m1();
        crate::law::positive_result(envelope.sl)?;
        crate::law::positive_result(envelope.sr)?;
        // Envelope masses: 1/sl and 1/sr for the exponential tails and
        // right - left for the flat centre; one uniform picks the piece.
        envelope.al = S::ONE / envelope.sl;
        envelope.ar = S::ONE / envelope.sr;
        envelope.area = finite(envelope.al + right - left + envelope.ar)?;
        Ok(envelope)
    }
    fn loss(&self, t: S) -> Result<S, SciOxideError> {
        Ok(self.a * expm1_minus(t)? + self.b * expm1_minus(-t)?)
    }
}
impl<S: RealScalar> GeneralizedInverseGaussian<S> {
    /// Creates a GIG distribution with index `p` and concentration `b`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `p` or `b` is NaN or infinite.
    /// - `DomainError` if `b ≤ 0`.
    /// - `NumericalInstability` if the normalizing constant `2 K_p(b)` is outside the range of `S`.
    /// - `ConvergenceFailure` if the Bessel function exhausts the work limit.
    pub fn new(p: S, b: S) -> Result<Self, SciOxideError> {
        Self::with_options(p, b, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(p: S, b: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(p, "p")?;
        argument(b, "b")?;
        if b <= S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("b"),
                "GIG requires b > 0",
            ));
        }
        let log_norm =
            finite(c::<S>(2.)?.ln() + crate::distribution::special::scaled_log_k(p, b, options)?)?;
        Self::from_normalizer(p, b, log_norm, options)
    }

    fn from_normalizer(
        p: S,
        b: S,
        log_norm: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        // Tails are integrated in y = ln x, where the density
        // exp(p y - b (cosh y - 1)) is log-concave with exponential tails. Its
        // mode solves p = b sinh(y), i.e. y = asinh(p / b) written with logs,
        // and 1 / sqrt(hypot(p, b)) is the curvature scale that sets the
        // quadrature panel width.
        let mode = if p == S::ZERO {
            S::ZERO
        } else {
            p.signum() * ((p.abs() + p.hypot(b)).ln() - b.ln())
        };
        let width = S::ONE / p.hypot(b).sqrt();
        Ok(Self {
            p,
            b,
            log_norm,
            mode,
            width: width.min(c(8.)?),
            envelope: GigEnvelope::new(p, b),
            options,
        })
    }
    fn coordinate_density(&self, y: S) -> Result<S, SciOxideError> {
        let sh = (y * S::HALF).sinh();
        if !sh.is_finite() {
            return Ok(S::NEG_INFINITY);
        }
        // Remove exp(-b) analytically; never subtract two O(b) normalizers.
        Ok(self.p * y - (self.b * sh) * sh * c::<S>(2.)? - self.log_norm)
    }
    fn log_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        let y = x.ln();
        // Evaluate the tail away from the mode and complement only a resolved large tail.
        if (y >= self.mode) == upper {
            tail(
                |z, _| self.coordinate_density(z),
                y,
                upper,
                self.mode,
                self.width,
                self.options,
                left,
            )
        } else {
            log1mexp(tail(
                |z, _| self.coordinate_density(z),
                y,
                !upper,
                self.mode,
                self.width,
                self.options,
                left,
            )?)
        }
    }
    fn moment(&self, k: S, left: &mut usize) -> Result<S, SciOxideError> {
        finite(
            (crate::distribution::special::scaled_log_k_with_work(self.p + k, self.b, left)?
                - self.log_norm
                + c::<S>(2.)?.ln())
            .exp(),
        )
    }
    // Rejection from the prepared log-concave envelope in log(x); 1 / GIG(p, b)
    // is GIG(-p, b), so draws use |p| and reflect at the end.
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let e = self.envelope.as_ref().map_err(Clone::clone)?;
        let mode = e.mode;
        let t = crate::util::rejection(|| {
            let u = crate::util::validated_open_float(rng)? * e.area;
            let v = crate::util::validated_open_float(rng)?;
            let (t, envelope) = if u < e.al {
                let d = v.ln() / e.sl;
                (e.left + d, e.sl * d)
            } else if u < e.area - e.ar {
                (e.left + v * (e.right - e.left), S::ZERO)
            } else {
                let d = -v.ln() / e.sr;
                (e.right + d, -e.sr * d)
            };
            Ok(
                (crate::util::validated_open_float(rng)?.ln() <= -e.loss(t)? - envelope)
                    .then_some(t),
            )
        })?;
        let log_x = if self.p < S::ZERO {
            -mode - t
        } else {
            mode + t
        };
        let x = finite(log_x.exp())?;
        if x <= S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "GIG sample underflow",
            ));
        }
        Ok(x)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedInverseGaussian<S> {
    const INVERSE: Inverse = Inverse::Lattice;
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
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            Ok(S::NEG_INFINITY)
        } else {
            Ok(self.coordinate_density(x.ln())? - x.ln())
        }
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_tail(x, upper, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.variate(rng)
    }
}
formula_law!(GeneralizedInverseGaussian);
impl<S: RealScalar> Mean<S> for GeneralizedInverseGaussian<S> {
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
impl<S: RealScalar> GeneralizedInverseGaussian<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        self.moment(S::ONE, left)
    }
}

impl<S: RealScalar> Variance<S> for GeneralizedInverseGaussian<S> {
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
impl<S: RealScalar> GeneralizedInverseGaussian<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let m = self.mean_work(left)?;
        crate::distribution::moment_math::variance_from_moments(self.moment(c(2.)?, left)?, m)
    }
}

/// Generalized hyperbolic distribution with index `p`, tail parameter `a` and skewness `b`.
///
/// With `γ = √(a² − b²)` and `h = √(1 + x²)`,
///
/// ```text
/// f(x) = γ^p h^(p−1/2) K_(p−1/2)(a h) e^(b x) / (√(2π) a^(p−1/2) K_p(γ)),
/// ```
///
/// where `K_ν` is the modified Bessel function of the second kind (DLMF 10.25).
/// The parameters satisfy `a > 0` and `|b| < a`, or `|b| = a` with `p < 0`. The
/// law has zero location and unit scale; shift and scale it with
/// [`Affine`](crate::Affine).
///
/// `X = bW + √W Z` is a normal variance-mean mixture with a standard normal `Z`
/// and `W ~ GIG(p, γ)/γ`. At `|b| = a` the mixing law `W` is inverse gamma.
/// Draws use this representation. Tail probabilities are computed by
/// quadrature in `asinh x`.
///
/// # References
///
/// O. Barndorff-Nielsen, "Exponentially decreasing distributions for the
/// logarithm of particle size", *Proc. R. Soc. Lond. A* 353, 1977, 401–419.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, GeneralizedHyperbolic, Quantile, SeedableRng, SplitMix64,
///     Variance,
/// };
/// let law = GeneralizedHyperbolic::<f64>::with_options(
///     1.0,
///     2.0,
///     0.5,
///     scioxide_random::LawOptions::default()
///         .with_work_budget(scioxide_core::Budget::ONE.saturating_add(1_048_575)),
/// )?;
/// let x = law.quantile(0.3)?;
/// assert!((law.cdf(x)? - 0.3).abs() < 1e-10);
/// assert!(law.variance()? > 0.0);
/// let mut rng = SplitMix64::from_seed(52);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedHyperbolic<S: RealScalar> {
    p: S,
    a: S,
    b: S,
    gamma: S,
    log_norm: S,
    mode: S,
    width: S,
    /// Prepared mixing variate; a preparation failure affects only sampling.
    mixing: Result<Mixing<S>, SciOxideError>,
    options: LawOptions<S>,
}

/// Mixing variable of the normal variance-mean representation.
#[derive(Debug, Clone)]
enum Mixing<S: RealScalar> {
    /// At `gamma = 0` the GIG degenerates to an inverse gamma `1 / Gamma(-p, 2)`.
    InverseGamma(GammaKernel<S>),
    /// `GIG(p, gamma) / gamma`.
    Gig(GeneralizedInverseGaussian<S>),
}
impl<S: RealScalar> GeneralizedHyperbolic<S> {
    /// Creates a generalized hyperbolic distribution with parameters `p`, `a` and `b`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `p`, `a` or `b` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0`, `|b| > a`, or `|b| = a` with `p ≥ 0`.
    /// - `NumericalInstability` if the normalizing constant is outside the range of `S`.
    /// - `ConvergenceFailure` if locating the mode or evaluating the Bessel
    ///   functions exhausts the work limit.
    pub fn new(p: S, a: S, b: S) -> Result<Self, SciOxideError> {
        Self::with_options(p, a, b, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(p: S, a: S, b: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        let left = &mut options.work_budget().get();
        options.validate()?;
        argument(p, "p")?;
        argument(a, "a")?;
        argument(b, "b")?;
        if a <= S::ZERO || b.abs() > a || (b.abs() == a && p >= S::ZERO) {
            return Err(crate::law::domain(
                if a <= S::ZERO {
                    scioxide_core::argument!("a")
                } else if b.abs() > a {
                    scioxide_core::argument!("b")
                } else {
                    // |b| = a is allowed only with p < 0.
                    scioxide_core::argument!("p")
                },
                "invalid generalized hyperbolic shapes",
            ));
        }
        // gamma = sqrt(a^2 - b^2), factored as sqrt(a - |b|) sqrt(2 (a + |b|) / 2)
        // so the boundary |b| -> a is resolved without cancellation or
        // overflow of the squares.
        let gamma =
            (a - b.abs()).sqrt() * (a * S::HALF + b.abs() * S::HALF).sqrt() * c::<S>(2.)?.sqrt();
        let log_mix = if gamma == S::ZERO {
            (p + S::ONE) * c::<S>(2.)?.ln() - scioxide_special::ln_gamma(-p)?
        } else {
            p * gamma.ln() - crate::distribution::special::scaled_log_k_with_work(p, gamma, left)?
        };
        let log_norm = finite(
            log_mix
                - crate::distribution::special::log_two_pi::<S>() * S::HALF
                - (p - S::HALF) * a.ln(),
        )?;
        let mixing = if gamma == S::ZERO {
            GammaKernel::new(-p).map(Mixing::InverseGamma)
        } else {
            GeneralizedInverseGaussian::from_normalizer(
                p,
                gamma,
                c::<S>(2.)?.ln() + p * gamma.ln() - log_mix,
                options,
            )
            .map(Mixing::Gig)
        };
        let mut law = Self {
            p,
            a,
            b,
            gamma,
            log_norm,
            mode: S::ZERO,
            width: S::ONE,
            mixing,
            options,
        };
        // Locate the peak in asinh(x), including the Jacobian. Bounded golden search.
        let mut lo = -S::ONE;
        let mut hi = S::ONE;
        for _ in 0..32 {
            let center = (lo + hi) * S::HALF;
            let fc = law.coordinate_density(center, left)?;
            if law.coordinate_density(lo, left)? > fc {
                lo *= c(2.)?;
            } else if law.coordinate_density(hi, left)? > fc {
                hi *= c(2.)?;
            } else {
                break;
            }
        }
        let ratio = c::<S>(0.3819660112501051)?;
        for _ in 0..64 {
            let l = lo + (hi - lo) * ratio;
            let r = hi - (hi - lo) * ratio;
            if law.coordinate_density(l, left)? < law.coordinate_density(r, left)? {
                lo = l;
            } else {
                hi = r;
            }
        }
        law.mode = (lo + hi) * S::HALF;
        // Width follows the local curvature, capped to avoid skipping exponential tails.
        let h = c::<S>(0.01)?;
        let peak = law.coordinate_density(law.mode, left)?;
        let curvature = (c::<S>(2.)? * peak
            - law.coordinate_density(law.mode - h, left)?
            - law.coordinate_density(law.mode + h, left)?)
            / (h * h);
        law.width = if curvature > S::ONE {
            S::ONE / curvature.sqrt()
        } else {
            S::ONE
        };
        Ok(law)
    }
    fn coordinate_density(&self, t: S, left: &mut usize) -> Result<S, SciOxideError> {
        let log_h = t.abs() - c::<S>(2.)?.ln();
        // Leading scaled-K asymptotic only once its order-dependent correction is below epsilon.
        if self.a.ln() + log_h
            > c::<S>(2.)? * (S::ONE + self.p.abs()).ln() - S::EPSILON.ln() + c::<S>(4.)?
        {
            let rate = self.a - self.b * t.signum();
            let decay = if rate == S::ZERO {
                S::ZERO
            } else {
                (rate.ln() + log_h).exp()
            };
            return Ok(self.log_norm
                + self.gamma
                + self.p * log_h
                + S::HALF * (S::PI.ln() - c::<S>(2.)?.ln() - self.a.ln())
                - decay);
        }
        let x = t.sinh();
        if !x.is_finite() {
            return Ok(S::NEG_INFINITY);
        }
        crate::law::charge(left)?;
        Ok(self.log_pdf_work(x, left)? + S::ONE.hypot(x).ln())
    }
    fn log_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        // Integrate in t = asinh(x): the exponential tails of the law become
        // doubly exponential in t, so the mapped half-line quadrature
        // converges quickly. As for GIG, only the tail away from the mode is
        // integrated; the other side is its complement.
        let t = x.asinh();
        if (t >= self.mode) == upper {
            tail(
                |z, left| self.coordinate_density(z, left),
                t,
                upper,
                self.mode,
                self.width,
                self.options,
                left,
            )
        } else {
            log1mexp(tail(
                |z, left| self.coordinate_density(z, left),
                t,
                !upper,
                self.mode,
                self.width,
                self.options,
                left,
            )?)
        }
    }
    fn mixing_moment(&self, k: S, left: &mut usize) -> Result<S, SciOxideError> {
        if self.gamma == S::ZERO {
            if -self.p <= k {
                return Err(crate::law::undefined_moment(
                    "hyperbolic mixing moment does not exist",
                ));
            }
            let k = k.into_f64();
            finite(c::<S>(
                (crate::distribution::special::log_gamma_ratio(-self.p.into_f64(), -k)?
                    - k * core::f64::consts::LN_2)
                    .exp(),
            )?)
        } else {
            finite(
                (log_k(self.p + k, self.gamma, left)?
                    - log_k(self.p, self.gamma, left)?
                    - k * self.gamma.ln())
                .exp(),
            )
        }
    }

    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let h = S::ONE.hypot(x);
        let z = finite(self.a * h)?;
        let log_scaled =
            crate::distribution::special::scaled_log_k_with_work(self.p - S::HALF, z, left)?;
        // a*h - b*x evaluated without subtracting two huge, nearly equal terms.
        let decay = (self.a - self.b.abs()) * x.abs()
            + self.a / (h + x.abs())
            + if self.b * x < S::ZERO {
                c::<S>(2.)? * self.b.abs() * x.abs()
            } else {
                S::ZERO
            };
        let denominator = self.a * h + self.b * x + self.gamma;
        let residual = self.gamma.mul_add(x, -self.b);
        let exponent = if self.gamma > S::ZERO
            && denominator >= self.a * h * S::HALF
            && denominator.is_finite()
            && residual.is_finite()
        {
            // (a*h-b*x)-gamma = (gamma*x-b)^2/(a*h+b*x+gamma).
            -(residual / denominator) * residual
        } else {
            self.gamma - decay
        };
        Ok(self.log_norm + (self.p - S::HALF) * h.ln() + log_scaled + exponent)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedHyperbolic<S> {
    const INVERSE: Inverse = Inverse::Lattice;
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
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, &mut self.options.work_budget().get())
    }
    fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, left)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_tail(x, upper, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // Normal variance-mean mixture: X = b W + sqrt(W) Z with a GIG mixing
        // variable W. At the gamma = 0 boundary the GIG degenerates to an
        // inverse gamma, which is sampled directly.
        let w = match self.mixing.as_ref().map_err(Clone::clone)? {
            Mixing::InverseGamma(kernel) => {
                S::ONE / scioxide_core::checked_float_product(S::TWO, kernel.sample(rng)?)?
            }
            Mixing::Gig(law) => law.variate(rng)? / self.gamma,
        };
        let z = crate::distribution::normal::standard_normal::<S, R>(rng)?;
        finite(self.b * w + w.sqrt() * z)
    }
}
formula_law!(GeneralizedHyperbolic);
impl<S: RealScalar> Mean<S> for GeneralizedHyperbolic<S> {
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
impl<S: RealScalar> GeneralizedHyperbolic<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        finite(self.b * self.mixing_moment(S::ONE, left)?)
    }
}

impl<S: RealScalar> Variance<S> for GeneralizedHyperbolic<S> {
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
impl<S: RealScalar> GeneralizedHyperbolic<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let m = self.mixing_moment(S::ONE, left)?;
        if self.b == S::ZERO {
            return finite(m);
        }
        let mixing_variance = crate::distribution::moment_math::variance_from_moments(
            self.mixing_moment(c(2.)?, left)?,
            m,
        )?;
        finite(m + self.b * self.b * mixing_variance)
    }
}

/// Normal inverse Gaussian distribution with tail parameter `a` and skewness `b`.
///
/// It is the [`GeneralizedHyperbolic`] law with `p = −1/2`. The parameters
/// satisfy `a > |b|`, and all moments exist.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, NormalInverseGaussian, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = NormalInverseGaussian::<f64>::with_options(
///     2.0,
///     1.0,
///     scioxide_random::LawOptions::default()
///         .with_work_budget(scioxide_core::Budget::ONE.saturating_add(1_048_575)),
/// )?;
/// // Mean b / √(a² − b²) = 1/√3.
/// assert!((law.mean()? - 1.0 / 3.0_f64.sqrt()).abs() < 1e-15);
/// let x = law.quantile(0.75)?;
/// assert!((law.cdf(x)? - 0.75).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(53);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct NormalInverseGaussian<S: RealScalar> {
    base: GeneralizedHyperbolic<S>,
}
impl<S: RealScalar> NormalInverseGaussian<S> {
    /// Creates a normal inverse Gaussian distribution with parameters `a` and `b`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `b` is NaN or infinite.
    /// - `DomainError` if `a ≤ |b|`.
    /// - The errors of [`GeneralizedHyperbolic::new`] with `p = −1/2`.
    pub fn new(a: S, b: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, b, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(a: S, b: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(a, "a")?;
        argument(b, "b")?;
        if a <= b.abs() {
            return Err(crate::law::domain(
                scioxide_core::argument!("a"),
                "normal inverse Gaussian requires a > |b|",
            ));
        }
        Ok(Self {
            base: GeneralizedHyperbolic::with_options(-S::HALF, a, b, options)?,
        })
    }
}
// The p = -1/2 generalized hyperbolic law: every capability is the base law's.
impl<S: RealScalar> Formula<S> for NormalInverseGaussian<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.base.tolerance()
    }
    fn work_budget(&self) -> scioxide_core::Budget {
        self.base.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.base.byte_limit()
    }
    fn bounds(&self) -> (Option<S>, Option<S>) {
        self.base.bounds()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        self.base.density_log(x)
    }
    fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.base.density_with_work(x, left)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        self.base.tail_with_work(x, upper, left)
    }
    fn log_probability_with_work(
        &self,
        x: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.base.log_probability_with_work(x, upper, left)
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.base.inverse_with_remaining(p, upper, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.base.draw_with_remaining(rng, left)
    }
}
formula_law!(NormalInverseGaussian);
impl<S: RealScalar> Mean<S> for NormalInverseGaussian<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(self.base.b / self.base.gamma)
    }
}
impl<S: RealScalar> Variance<S> for NormalInverseGaussian<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let r = self.base.a / self.base.gamma;
        finite(r * r / self.base.gamma)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use crate::{SeedableRng, SplitMix64};
    #[test]
    fn gig_draw_underflow_is_not_invalid_input() {
        let b = f64::from_bits(1);
        // Supply the finite log normalizer independently of the Bessel kernel's
        // range limit, so the negative-shape draw reaches its own underflow check.
        let mode = -(2000.0_f64.ln() - b.ln());
        let law = GeneralizedInverseGaussian {
            p: -1000.0,
            b,
            log_norm: scioxide_special::ln_gamma(1000.0).unwrap()
                + 1000.0 * (core::f64::consts::LN_2 - b.ln()),
            mode,
            width: 1.0 / 1000.0_f64.sqrt(),
            envelope: GigEnvelope::new(-1000.0, b),
            options: LawOptions::<f64>::default(),
        };
        crate::error_assertions::numerical(
            law.variate(&mut SplitMix64::from_seed(7)).unwrap_err(),
            scioxide_core::NumericalIssue::Underflow,
        );
    }
}
