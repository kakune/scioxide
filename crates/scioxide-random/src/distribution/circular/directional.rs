use super::angular::AngleKernel;
use super::*;
use crate::BitGenerator;
use crate::distribution::LawOptions;
use crate::distribution::rows::{RowLaw, Scratch, vector_sampling};
use alloc::vec::Vec;

fn dimension<S: RealScalar>(
    d: usize,
    options: LawOptions<S>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if d == 0 {
        return Err(invalid(argument, "sphere dimension must be positive"));
    }
    options.admit::<S>(d)?;
    if d > options.work.get() {
        return Err(exhausted());
    }
    Ok(())
}
fn shape(d: usize, n: usize) -> Result<(), SciOxideError> {
    if d != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(d),
            scioxide_core::Dimensions::one(n),
        ));
    }
    Ok(())
}
fn membership<S: RealScalar>(d: usize, x: &[S]) -> Result<bool, SciOxideError> {
    shape(d, x.len())?;
    let mut sum = 0.0;
    for &v in x {
        argument(v, "x")?;
        let v = v.into_f64();
        if v.abs() > 1.0 + 8.0 * S::EPSILON.into_f64() {
            return Ok(false);
        }
        sum += v * v;
    }
    Ok(sum > 0.0 && (sum - 1.0).abs() <= 8.0 * d as f64 * S::EPSILON.into_f64())
}
fn log_area(d: usize) -> Result<f64, SciOxideError> {
    let half = 0.5 * d as f64;
    Ok(core::f64::consts::LN_2 + half * PI.ln() - scioxide_special::ln_gamma(half)?)
}
// One Gaussian normalization kernel, also used for vMF tangent directions.
fn uniform<S: RealScalar, R: Rng<f64>>(rng: &mut R, out: &mut [S]) -> Result<(), SciOxideError> {
    if out.len() == 1 {
        out[0] = number(if rng.next_float()? < 0.5 { -1.0 } else { 1.0 })?;
        return Ok(());
    }
    let normal = crate::Normal::<f64>::standard();
    let norm = crate::util::rejection(|| {
        let mut norm = 0.0_f64;
        for value in &mut *out {
            let x = normal.sample(rng)?;
            *value = number(x)?;
            norm = norm.hypot(value.into_f64());
        }
        Ok((norm > 0.0).then_some(norm))
    })?;
    for value in &mut *out {
        *value = number(value.into_f64() / norm)?;
    }
    Ok(())
}

/// Uniform distribution on the unit sphere `S^(d−1) = {x ∈ ℝᵈ : ‖x‖₂ = 1}`, `d ≥ 1`.
///
/// The density with respect to surface measure is `Γ(d/2) / (2 π^(d/2))`, the
/// reciprocal of the sphere's area. A draw normalizes a vector of `d`
/// independent standard normals. For `d = 1` the law is uniform on `{−1, 1}`.
///
/// ```
/// use scioxide_random::{Density, SeedableRng, SplitMix64, UniformDirection};
/// let law = UniformDirection::<f64>::new(3)?;
/// assert!((law.pdf(&[0.0, 0.0, 1.0])? - 1.0 / (4.0 * core::f64::consts::PI)).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(74);
/// let direction = law.sample(&mut rng)?;
/// assert!(law.contains_support(&direction)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct UniformDirection<S: RealScalar> {
    dimension: usize,
    log_density: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> UniformDirection<S> {
    /// Creates the law on the unit sphere in `ℝᵈ`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `d = 0`.
    /// - `ResourceLimit` if `d` values exceed [`LawOptions::max_bytes`].
    /// - `ConvergenceFailure` if `d` exceeds [`LawOptions::work_budget`].
    pub fn new(d: usize) -> Result<Self, SciOxideError> {
        Self::with_options(d, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(d: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        dimension(d, options, scioxide_core::argument!("d"))?;
        Ok(Self {
            dimension: d,
            log_density: -log_area(d)?,
            options,
        })
    }
    /// Returns the dimension `d`, the number of elements in a sample.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether `x` is a unit vector: `|‖x‖₂² − 1| ≤ 8 d ε`, with `ε` the
    /// machine epsilon of `S`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    pub fn contains_support(&self, x: &[S]) -> Result<bool, SciOxideError> {
        membership(self.dimension, x)
    }
}
impl<S: RealScalar> RowLaw<S, S> for UniformDirection<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        _: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        uniform(
            &mut crate::distribution::words::Words::new(rng, self.options.sample_words.get()),
            out,
        )
    }
}
impl<S: RealScalar> LogDensity<S, &[S]> for UniformDirection<S> {
    fn log_pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        number(self.log_density)
    }
}
impl<S: RealScalar> Density<S, &[S]> for UniformDirection<S> {
    fn pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        finite(self.log_pdf(x)?.exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for UniformDirection<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        scioxide_core::backend::try_filled(
            self.dimension,
            S::ZERO,
            scioxide_core::quantity!("sphere_mean"),
        )
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for UniformDirection<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        scioxide_core::backend::try_filled(
            self.dimension,
            number(1.0 / self.dimension as f64)?,
            scioxide_core::quantity!("sphere_variance"),
        )
    }
}
impl<S: RealScalar> Entropy<S> for UniformDirection<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        number(-self.log_density)
    }
}

/// Von Mises–Fisher distribution on the unit sphere in `ℝᵈ`, `d ≥ 2`, with mean
/// direction `μ` and concentration `κ ≥ 0`.
///
/// The density with respect to surface measure is
///
/// ```text
/// f(x) = κ^(d/2−1) / ((2π)^(d/2) I_(d/2−1)(κ)) · exp(κ μᵀx)
/// ```
///
/// where `I_ν` is the modified Bessel function of the first kind. `κ = 0` is the
/// uniform distribution on the sphere. Draws use the [`VonMises`](crate::VonMises)
/// angle sampler for `d = 2`, inversion of the exact component law for `d = 3`,
/// and Wood's (1994) rejection method for `d ≥ 4`.
///
/// ```
/// use scioxide_random::{Density, Mean, SeedableRng, SplitMix64, VonMisesFisher};
/// let law = VonMisesFisher::<f64>::new(&[0.0, 0.0, 1.0], 5.0)?;
/// assert!(law.pdf(&[0.0, 0.0, 1.0])? > law.pdf(&[0.0, 0.0, -1.0])?);
/// let length = 1.0 / 5.0_f64.tanh() - 0.2;
/// assert!((law.mean_resultant_length()? - length).abs() < 1e-12);
/// assert!((law.mean()?[2] - length).abs() < 1e-12);
/// let mut rng = SplitMix64::from_seed(75);
/// assert!(law.contains_support(&law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. T. A. Wood, "Simulation of the von Mises Fisher distribution",
/// *Commun. Statist. Simul. Comput.* 23(1), 1994, 157–164.
#[derive(Debug, Clone)]
pub struct VonMisesFisher<S: RealScalar> {
    mean: Vec<S>,
    mean_norm: f64,
    kappa: f64,
    log_peak: f64,
    resultant: f64,
    deficit: f64,
    draw: DirectionKernel,
    options: LawOptions<S>,
}
#[derive(Debug, Clone)]
enum DirectionKernel {
    Uniform,
    Circle(AngleKernel),
    Sphere,
    Wood {
        beta: crate::distribution::beta::BetaKernel<f64>,
        b: f64,
        delta0: f64,
    },
}
impl<S: RealScalar> VonMisesFisher<S> {
    /// Creates the law with unit mean direction `mean` and `κ = concentration`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `mean.len() < 2`.
    /// - `NonFinite` if `mean` or `concentration` has a NaN or infinite value.
    /// - `DomainError` if `mean` is not a unit vector (see [`Self::contains_support`])
    ///   or `concentration < 0`.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or the Bessel function
    ///   evaluation exceeds the [work limits](crate#work-limits).
    pub fn new(mean: &[S], concentration: S) -> Result<Self, SciOxideError> {
        Self::with_options(mean, concentration, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        mean: &[S],
        concentration: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let d = mean.len();
        dimension(d, options, scioxide_core::argument!("mean"))?;
        if d < 2 {
            return Err(invalid(
                scioxide_core::argument!("mean"),
                "von Mises-Fisher requires dimension at least two",
            ));
        }
        if !membership(d, mean)? {
            return Err(crate::law::domain(
                scioxide_core::argument!("mean"),
                "von Mises-Fisher requires a unit mean direction",
            ));
        }
        let kappa =
            nonnegative(concentration, scioxide_core::argument!("concentration"))?.into_f64();
        let nu = 0.5 * d as f64 - 1.0;
        let (log_peak, resultant, deficit) = if kappa == 0.0 {
            (-log_area(d)?, 0.0, 1.0)
        } else if kappa < 1.0 || kappa < nu {
            // Normalized I_nu series avoids underflow of k^nu/Gamma(nu+1).
            let mut left = options.work.get();
            let a = log_series(nu, kappa, options.relative(), &mut left)?;
            let b = log_series(nu + 1.0, kappa, options.relative(), &mut left)?;
            let resultant = kappa / (d as f64) * (b - a).exp();
            (-log_area(d)? + kappa - a, resultant, 1.0 - resultant)
        } else {
            let mut left = options.work.get();
            let a = scaled_bessel(nu, kappa, &mut left)?;
            let b = scaled_bessel(nu + 1.0, kappa, &mut left)?;
            crate::law::positive_result(a)?;
            let deficit = if kappa > 1e8 * (d as f64).powi(2) {
                // DLMF 10.40 ratio expansion; retain k*(1-A) when A rounds to 1.
                (d as f64 - 1.0) / (2.0 * kappa) * (1.0 - (d as f64 - 3.0) / (4.0 * kappa))
            } else {
                1.0 - b / a
            };
            (
                nu * kappa.ln()
                    - 0.5 * d as f64 * crate::distribution::special::log_two_pi::<f64>()
                    - a.ln(),
                1.0 - deficit,
                deficit,
            )
        };
        let mut owned = scioxide_core::backend::try_with_capacity(
            d,
            scioxide_core::quantity!("direction_parameters"),
        )?;
        owned.extend_from_slice(mean);
        let draw = if kappa == 0.0 {
            DirectionKernel::Uniform
        } else if d == 2 {
            DirectionKernel::Circle(AngleKernel::new(kappa))
        } else if d == 3 {
            DirectionKernel::Sphere
        } else {
            let m = (d - 1) as f64;
            let ratio = 2.0 * (kappa / m);
            let b = if ratio <= 1.0 {
                1.0 / (ratio.hypot(1.0) + ratio)
            } else {
                let inverse = (m / kappa) * 0.5;
                inverse / (inverse.hypot(1.0) + 1.0)
            };
            DirectionKernel::Wood {
                beta: crate::distribution::beta::BetaKernel::new(m * 0.5, m * 0.5)?,
                b,
                delta0: 2.0 * b / (1.0 + b),
            }
        };
        Ok(Self {
            mean: owned,
            mean_norm: mean
                .iter()
                .fold(0.0_f64, |norm, x| norm.hypot(x.into_f64())),
            kappa,
            log_peak,
            resultant,
            deficit,
            draw,
            options,
        })
    }
    /// Returns the dimension `d`, the number of elements in a sample.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }
    /// Returns the mean direction `μ` as given, also when `κ = 0`.
    pub fn direction(&self) -> &[S] {
        &self.mean
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether `x` is a unit vector: `|‖x‖₂² − 1| ≤ 8 d ε`, with `ε` the
    /// machine epsilon of `S`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    pub fn contains_support(&self, x: &[S]) -> Result<bool, SciOxideError> {
        membership(self.dimension(), x)
    }
    /// Returns the mean resultant length `‖E[X]‖ = I_(d/2)(κ) / I_(d/2−1)(κ)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value is outside the range of `S`.
    pub fn mean_resultant_length(&self) -> Result<S, SciOxideError> {
        number(self.resultant)
    }
}
impl<S: RealScalar> RowLaw<S, S> for VonMisesFisher<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        _: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let d = self.dimension();
        if matches!(self.draw, DirectionKernel::Uniform) {
            return uniform(&mut words, out);
        }
        if let DirectionKernel::Circle(angle) = &self.draw {
            let theta = angle.sample(&mut words)?;
            let (s, c) = theta.sin_cos();
            let a = self.mean[0].into_f64() / self.mean_norm;
            let b = self.mean[1].into_f64() / self.mean_norm;
            out[0] = number(a * c - b * s)?;
            out[1] = number(b * c + a * s)?;
            return Ok(());
        }
        // Keep 1-w separately so high concentration retains tangent variation.
        let delta = if d == 3 {
            let u: f64 = crate::util::validated_open_float(&mut words)?;
            if self.kappa < 0.5 {
                -((1.0 - u) * (-2.0 * self.kappa).exp_m1()).ln_1p() / self.kappa
            } else {
                -crate::law::logadd(u.ln(), (-u).ln_1p() - 2.0 * self.kappa)? / self.kappa
            }
        } else {
            let m = (d - 1) as f64;
            let DirectionKernel::Wood { beta, b, delta0 } = &self.draw else {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "missing directional sampling kernel",
                ));
            };
            let (b, delta0) = (*b, *delta0);
            crate::util::rejection(|| {
                let z = beta.sample(&mut words)?;
                let denominator = 1.0 - z + b * z;
                let delta = 2.0 * b * z / denominator;
                let log_ratio = ((1.0 + b) * 0.5 * (1.0 + (1.0 - b) * z / denominator)).ln();
                let bound = self.kappa * (delta0 - delta) + m * log_ratio;
                let u: f64 = crate::util::validated_open_float(&mut words)?;
                Ok((u.ln() <= bound).then_some(delta))
            })?
        };
        uniform(&mut words, &mut out[1..])?;
        let tangent = (delta * (2.0 - delta)).max(0.0).sqrt();
        for value in &mut out[1..] {
            *value = number(value.into_f64() * tangent)?;
        }
        out[0] = number(1.0 - delta)?;
        // Stable Householder: choose the source pole opposite mean[0], so the
        // reflector denominator cannot cancel near either coordinate pole.
        let sign = if self.mean[0] >= S::ZERO { -1.0 } else { 1.0 };
        out[0] = number(sign * out[0].into_f64())?;
        let v0 = sign - (self.mean[0].into_f64() / self.mean_norm);
        let dot = v0 * out[0].into_f64()
            - self.mean[1..]
                .iter()
                .zip(&out[1..])
                .map(|(a, b)| (a.into_f64() / self.mean_norm) * b.into_f64())
                .sum::<f64>();
        let factor = dot / (1.0 - sign * (self.mean[0].into_f64() / self.mean_norm));
        out[0] = number(out[0].into_f64() - factor * v0)?;
        for (value, mu) in out[1..].iter_mut().zip(&self.mean[1..]) {
            *value = number(value.into_f64() + factor * (mu.into_f64() / self.mean_norm))?;
        }
        Ok(())
    }
}
fn log_series(nu: f64, kappa: f64, tolerance: f64, left: &mut usize) -> Result<f64, SciOxideError> {
    let mut sum = 0.0;
    let mut term = 0.0;
    let mut j = 0usize;
    loop {
        *left = left.checked_sub(1).ok_or_else(exhausted)?;
        j += 1;
        term += 2.0 * kappa.ln()
            - core::f64::consts::LN_2 * 2.0
            - (j as f64).ln()
            - (nu + j as f64).ln();
        sum = crate::law::logadd(sum, term)?;
        if term - sum < tolerance.ln() && (j as f64) * (nu + j as f64) > 0.25 * kappa * kappa {
            return Ok(sum);
        }
    }
}
impl<S: RealScalar> LogDensity<S, &[S]> for VonMisesFisher<S> {
    fn log_pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        // Squared chord distance retains tiny departures from the mode.
        let norm = x.iter().fold(0.0_f64, |norm, x| norm.hypot(x.into_f64()));
        let distance = x
            .iter()
            .zip(&self.mean)
            .map(|(a, b)| (a.into_f64() / norm - b.into_f64() / self.mean_norm).powi(2))
            .sum::<f64>();
        number(finite(self.log_peak - self.kappa * (0.5 * distance))?)
    }
}
impl<S: RealScalar> Density<S, &[S]> for VonMisesFisher<S> {
    fn pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        finite(self.log_pdf(x)?.exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for VonMisesFisher<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        let mut out = scioxide_core::backend::try_with_capacity(
            self.dimension(),
            scioxide_core::quantity!("direction_mean"),
        )?;
        for mu in &self.mean {
            out.push(number(self.resultant * (mu.into_f64() / self.mean_norm))?);
        }
        Ok(out)
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for VonMisesFisher<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        let d = self.dimension() as f64;
        let transverse = if self.kappa == 0.0 {
            1.0 / d
        } else {
            self.resultant / self.kappa
        };
        let axial = if self.kappa > 1e8 * d * d {
            (d - 1.0) / (2.0 * self.kappa) / self.kappa
        } else {
            self.deficit * (2.0 - self.deficit) - (d - 1.0) * transverse
        };
        let mut out = scioxide_core::backend::try_with_capacity(
            self.dimension(),
            scioxide_core::quantity!("direction_variance"),
        )?;
        for mu in &self.mean {
            let square = (mu.into_f64() / self.mean_norm).powi(2);
            out.push(number(
                (transverse * (1.0 - square) + axial * square).max(0.0),
            )?);
        }
        Ok(out)
    }
}
impl<S: RealScalar> Entropy<S> for VonMisesFisher<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        number(-self.log_peak + self.kappa * self.deficit)
    }
}

vector_sampling!(UniformDirection, S, S::ZERO);
vector_sampling!(VonMisesFisher, S, S::ZERO);
