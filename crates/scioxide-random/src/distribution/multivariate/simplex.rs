use super::*;
use crate::Entropy;
use crate::distribution::LawOptions;
use crate::distribution::gamma::GammaKernel;

// Analytic parameters are also used by the count law, which needs no Gamma cache.
#[derive(Debug, Clone)]
pub(super) struct SimplexParameters<S: RealScalar> {
    pub(super) alpha: Vec<f64>,
    pub(super) sum: f64,
    pub(super) normalizer: f64,
    pub(super) options: LawOptions<S>,
}
impl<S: RealScalar> SimplexParameters<S> {
    pub(super) fn new(alpha: &[S], options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        if alpha.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("alpha"),
                "Dirichlet needs nonempty concentrations",
            ));
        }
        options.admit::<f64>(alpha.len())?;
        options.steps(alpha.len().checked_mul(alpha.len()).ok_or_else(|| {
            invalid(scioxide_core::argument!("alpha"), "simplex work overflow")
        })?)?;
        let alpha = values(alpha)?;
        let mut sum = scioxide_core::RangeAccumulator::new();
        for &a in &alpha {
            sum.add(positive(a, scioxide_core::argument!("alpha"))?)?;
        }
        let sum = sum.finish()?;
        // ln Gamma(sum alpha) - sum ln Gamma(alpha_i) written as a telescoping
        // sum of -ln B(alpha_i, alpha_{i+1} + ... + alpha_d). Each Beta term is
        // well conditioned, whereas the Gamma form cancels heavily when some
        // concentrations are huge.
        let normalizer = (0..alpha.len() - 1).try_fold(0.0, |s, i| {
            Ok::<_, SciOxideError>(
                s - scioxide_special::ln_beta(alpha[i], alpha[i + 1..].iter().sum::<f64>())?,
            )
        })?;
        Ok(Self {
            alpha,
            sum,
            normalizer: finite(normalizer)?,
            options,
        })
    }
    pub(super) fn dimension(&self) -> usize {
        self.alpha.len()
    }
}

/// Dirichlet distribution with concentrations `α₁, …, α_d > 0` on the probability
/// simplex `{x : xᵢ ≥ 0, Σ xᵢ = 1}`.
///
/// With `α₀ = Σ αᵢ`, the density with respect to Lebesgue measure on the first
/// `d − 1` coordinates is
///
/// ```text
/// p(x) = Γ(α₀) / Πᵢ Γ(αᵢ) · Πᵢ xᵢ^(αᵢ − 1)
/// ```
///
/// and the mean is `αᵢ / α₀`. A draw normalizes `d` independent Gamma(αᵢ, 1)
/// variates. When some `αᵢ < 1`, it normalizes them from their logarithms, so
/// very small concentrations do not underflow every coordinate to zero.
///
/// ```
/// use scioxide_random::{Density, Dirichlet, Mean, SeedableRng, SplitMix64};
/// let law = Dirichlet::new(&[1.0_f64, 2.0, 3.0])?;
/// // Gamma(6) / (Gamma(1) Gamma(2) Gamma(3)) * 0.3 * 0.5^2 = 60 * 0.075
/// assert!((law.pdf(&[0.2, 0.3, 0.5])? - 4.5).abs() < 1e-12);
/// assert!((law.mean()?[2] - 0.5).abs() < 1e-15);
/// let x = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&x)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Dirichlet<S: RealScalar> {
    parameters: SimplexParameters<S>,
    kernels: Vec<GammaKernel<f64>>,
    ordinary: bool,
}
impl<S: RealScalar> Dirichlet<S> {
    /// Creates the law with concentrations `alpha`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `alpha` is empty.
    /// - `NonFinite` if a concentration is NaN or infinite.
    /// - `DomainError` if a concentration is not positive.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new(alpha: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(alpha, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(alpha: &[S], options: LawOptions<S>) -> Result<Self, SciOxideError> {
        let parameters = SimplexParameters::new(alpha, options)?;
        let count = if alpha.len() == 1 { 0 } else { alpha.len() };
        let bytes = scioxide_core::backend::checked_add(
            crate::workspace::storage_bytes::<f64>(parameters.alpha.capacity())?,
            crate::workspace::storage_bytes::<GammaKernel<f64>>(count)?,
            scioxide_core::quantity!("dirichlet_parameters"),
        )?;
        crate::workspace::admit_bytes(bytes, options.max_bytes())?;
        let mut kernels = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("dirichlet_kernels"),
        )?;
        for &a in parameters.alpha.iter().take(count) {
            kernels.push(GammaKernel::new(a)?);
        }
        let law = Self {
            ordinary: parameters.alpha.iter().all(|&a| a >= 1.0),
            parameters,
            kernels,
        };
        crate::workspace::admit_bytes(law.parameter_bytes()?, options.max_bytes())?;
        Ok(law)
    }
    fn parameter_bytes(&self) -> Result<usize, SciOxideError> {
        scioxide_core::backend::checked_add(
            crate::workspace::storage_bytes::<f64>(self.parameters.alpha.capacity())?,
            crate::workspace::storage_bytes::<GammaKernel<f64>>(self.kernels.capacity())?,
            scioxide_core::quantity!("dirichlet_parameters"),
        )
    }
    /// Returns the number of coordinates `d`.
    pub fn dimension(&self) -> usize {
        self.parameters.alpha.len()
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.parameters.options
    }
    /// Tests whether `x` lies on the simplex: `0 ≤ xᵢ ≤ 1` and `|Σ xᵢ − 1| ≤ 8 d ε`,
    /// with `ε` the machine epsilon of `S`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    pub fn contains_support(&self, x: &[S]) -> Result<bool, SciOxideError> {
        size(x.len(), self.dimension())?;
        let mut sum = 0.0;
        for &v in x {
            crate::law::argument(v, "x")?;
            let v = v.into_f64();
            if !(0.0..=1.0).contains(&v) {
                return Ok(false);
            }
            sum += v;
        }
        Ok((sum - 1.0).abs() <= 8.0 * x.len() as f64 * S::EPSILON.into_f64())
    }
    /// Returns the covariance matrix `(diag(p) − p pᵀ) / (α₀ + 1)`, `pᵢ = αᵢ / α₀`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the `d × d` result cannot be stored.
    /// - `NumericalInstability` if an entry is outside the range of `S`.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        self.parameters.options.admit::<S>(
            self.dimension()
                .checked_mul(self.dimension())
                .ok_or_else(|| {
                    invalid(scioxide_core::argument!("self"), "covariance size overflow")
                })?,
        )?;
        let p = converted::<f64>(
            self.parameters
                .alpha
                .iter()
                .map(|a| a / self.parameters.sum),
        )?;
        covariance(&p, 1.0 / (self.parameters.sum + 1.0))
    }
}
impl<S: RealScalar> RowLaw<S, S> for Dirichlet<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.parameters.options
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.parameter_bytes()
    }
    // Binary64 Gamma draws, including for binary32 output.
    fn scratch(&self) -> (usize, usize) {
        (0, self.kernels.len())
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        scratch: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let scratch = scratch.reals;
        self.parameters.options.steps(self.dimension())?;
        if self.dimension() == 1 {
            out[0] = S::ONE;
            return Ok(());
        }
        let mut words =
            crate::distribution::words::Words::new(rng, self.parameters.options.sample_words.get());
        if self.ordinary {
            // Shapes >= 1 need no underflow-prone power boost. Scale before
            // summing so even a sum of individually finite variates cannot overflow.
            let mut scale = 0.0_f64;
            for (value, kernel) in scratch.iter_mut().zip(&self.kernels) {
                *value = crate::law::positive_result(kernel.sample(&mut words)?)?;
                scale = scale.max(*value);
            }
            let norm = scratch.iter().map(|&value| value / scale).sum::<f64>();
            for (value, &sample) in out.iter_mut().zip(scratch.iter()) {
                *value = number((sample / scale) / norm)?;
            }
        } else {
            // Tiny Gamma variates may underflow; their widened logarithms still
            // distinguish concentrations before the final output conversion.
            let mut maximum = f64::NEG_INFINITY;
            for (value, kernel) in scratch.iter_mut().zip(&self.kernels) {
                *value = kernel.sample_log(&mut words)?;
                maximum = maximum.max(*value);
            }
            // Subtract the maximum before exponentiation: adding ln(d) to a
            // huge negative logarithm could round it away and break normalization.
            let norm = scratch
                .iter_mut()
                .map(|value| {
                    *value = (*value - maximum).exp();
                    *value
                })
                .sum::<f64>();
            for (value, &sample) in out.iter_mut().zip(scratch.iter()) {
                *value = number(sample / norm)?;
            }
        }
        Ok(())
    }
}
vector_sampling!(Dirichlet, S, S::ZERO);
impl<S: RealScalar> LogDensity<S, &[S]> for Dirichlet<S> {
    fn log_pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        let mut log = self.parameters.normalizer;
        let mut magnitude = log.abs();
        for (&a, &x) in self.parameters.alpha.iter().zip(x) {
            if a != 1.0 {
                let term = (a - 1.0) * x.into_f64().ln();
                magnitude += term.abs();
                log += term;
            }
        }
        if x.iter().all(|v| *v > S::ZERO) {
            return conditioned_number(log, magnitude);
        }
        if log == f64::INFINITY {
            Ok(S::INFINITY)
        } else {
            number(log)
        }
    }
}
impl<S: RealScalar> Density<S, &[S]> for Dirichlet<S> {
    fn pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        {
            let log = self.log_pdf(x)?.into_f64();
            if log == f64::INFINITY {
                Ok(S::INFINITY)
            } else {
                number(log.exp())
            }
        }
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for Dirichlet<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        converted(
            self.parameters
                .alpha
                .iter()
                .map(|a| a / self.parameters.sum),
        )
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for Dirichlet<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.parameters.alpha.iter().map(|a| {
            let p = a / self.parameters.sum;
            p * (1.0 - p) / (self.parameters.sum + 1.0)
        }))
    }
}
impl<S: RealScalar> Entropy<S> for Dirichlet<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        number(
            -self.parameters.normalizer
                + (self.parameters.sum - self.dimension() as f64)
                    * scioxide_special::digamma(self.parameters.sum)?
                - self.parameters.alpha.iter().try_fold(0.0, |s, &a| {
                    Ok::<_, SciOxideError>(s + (a - 1.0) * scioxide_special::digamma(a)?)
                })?,
        )
    }
}
