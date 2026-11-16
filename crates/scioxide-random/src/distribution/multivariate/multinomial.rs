//! Multinomial distribution.
use super::{RowLaw, Scratch, vector_sampling};
use crate::distribution::LawOptions;
use crate::distribution::discrete_math::PoissonMass;
use crate::law::{finite, invalid};
use crate::{BitGenerator, LogMass, Mass, Mean, Variance};
use alloc::vec::Vec;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

/// Multinomial distribution: the category counts of `n` independent trials with
/// category probabilities `p₁, …, p_k`.
///
/// The mass of `x` with `Σ xᵢ = n` is `n! Πᵢ pᵢ^xᵢ / xᵢ!`, and the mean is `n pᵢ`.
/// The probabilities must sum to 1 within `8 k ε`, with `ε` the machine epsilon
/// of `S`, and are then divided by their sum. A category with `pᵢ = 0` never
/// occurs. A draw takes one conditional binomial draw per category.
///
/// ```
/// use scioxide_random::{Mass, Mean, Multinomial, SeedableRng, SplitMix64};
/// let law = Multinomial::new(4, &[0.2_f64, 0.3, 0.5])?;
/// // 4! / (1! 1! 2!) * 0.2 * 0.3 * 0.5^2 = 0.18
/// assert!((law.pmf(&[1, 1, 2])? - 0.18).abs() < 1e-12);
/// assert!((law.mean()?[2] - 2.0).abs() < 1e-12);
/// let counts = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&counts)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Multinomial<S: RealScalar> {
    trials: usize,
    probabilities: Vec<S>,
    conditional: Vec<(f64, bool)>,
    masses: Vec<PoissonMass>,
    log_normalizer: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> Multinomial<S> {
    /// Creates the law with `n = trials` and category probabilities `probabilities`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `probabilities` is empty, if a positive probability
    ///   underflows to zero after normalization.
    /// - `NonFinite` if a probability is NaN or infinite.
    /// - `DomainError` if a probability is outside `[0, 1]` or the sum is not 1 within `8 k ε`.
    /// - `NotImplemented` if `trials > 2⁵³`.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new(trials: usize, probabilities: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(trials, probabilities, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        trials: usize,
        probabilities: &[S],
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        super::super::count_math::analytic_count(trials)?;
        if probabilities.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("probabilities"),
                "multinomial probabilities are empty",
            ));
        }
        let bytes = scioxide_core::backend::checked_add(
            crate::workspace::storage_bytes::<S>(probabilities.len())?,
            scioxide_core::backend::checked_add(
                crate::workspace::storage_bytes::<(f64, bool)>(probabilities.len())?,
                crate::workspace::storage_bytes::<PoissonMass>(probabilities.len())?,
                scioxide_core::quantity!("multinomial_parameters"),
            )?,
            scioxide_core::quantity!("multinomial_parameters"),
        )?;
        options.admit::<u8>(bytes)?;
        options.steps(probabilities.len())?;
        let mut sum = scioxide_core::RangeAccumulator::new();
        for &p in probabilities {
            crate::law::probability(p, scioxide_core::argument!("probabilities"))?;
            sum.add(p.into_f64())?;
        }
        let total = sum.finish()?;
        if total <= 0.0
            || (total - 1.0).abs() > 8.0 * probabilities.len() as f64 * S::EPSILON.into_f64()
        {
            return Err(crate::law::domain(
                scioxide_core::argument!("probabilities"),
                "probabilities must sum to one",
            ));
        }
        let mut values = scioxide_core::backend::try_with_capacity(
            probabilities.len(),
            scioxide_core::quantity!("multinomial_probabilities"),
        )?;
        for &p in probabilities {
            let normalized = S::try_from_f64_lossy(p.into_f64() / total)?;
            if p > S::ZERO && normalized == S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("probabilities"),
                    "normalized probability underflows",
                ));
            }
            values.push(normalized);
        }
        let mut conditional = scioxide_core::backend::try_with_capacity(
            values.len(),
            scioxide_core::quantity!("multinomial_conditionals"),
        )?;
        let mut rest = 0.0;
        for p in values.iter().rev() {
            let a = p.into_f64();
            let total = a + rest;
            conditional.push((
                if total == 0.0 {
                    0.0
                } else {
                    a.min(rest) / total
                },
                a > rest,
            ));
            rest = total;
        }
        conditional.reverse();
        let mut masses = scioxide_core::backend::try_with_capacity(
            values.len(),
            scioxide_core::quantity!("multinomial_masses"),
        )?;
        masses.extend(
            values
                .iter()
                .map(|p| PoissonMass::new(trials as f64 * p.into_f64())),
        );
        let law = Self {
            trials,
            probabilities: values,
            conditional,
            masses,
            log_normalizer: PoissonMass::new(trials as f64).log_mass(trials as f64),
            options,
        };
        crate::workspace::admit_bytes(law.parameter_bytes()?, options.max_bytes())?;
        Ok(law)
    }
    fn parameter_bytes(&self) -> Result<usize, SciOxideError> {
        use crate::workspace::storage_bytes;
        let q = scioxide_core::quantity!("multinomial_parameters");
        scioxide_core::backend::checked_add(
            storage_bytes::<S>(self.probabilities.capacity())?,
            scioxide_core::backend::checked_add(
                storage_bytes::<(f64, bool)>(self.conditional.capacity())?,
                storage_bytes::<PoissonMass>(self.masses.capacity())?,
                q,
            )?,
            q,
        )
    }
    /// Returns the number of trials `n`.
    pub fn trials(&self) -> usize {
        self.trials
    }
    /// Returns the number of categories `k`, including those with probability zero.
    pub fn categories(&self) -> usize {
        self.probabilities.len()
    }
    /// Returns the category probabilities after division by their sum.
    pub fn probabilities(&self) -> &[S] {
        &self.probabilities
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether `Σ xᵢ = n` and `xᵢ = 0` wherever `pᵢ = 0`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x.len() ≠ k` or `Σ xᵢ` overflows `usize`.
    pub fn contains_support(&self, x: &[usize]) -> Result<bool, SciOxideError> {
        if x.len() != self.categories() {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "multinomial count dimension mismatch",
            ));
        }
        let total = x.iter().try_fold(0usize, |s, &k| {
            s.checked_add(k).ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("x"),
                    "multinomial count sum overflow",
                )
            })
        })?;
        Ok(total == self.trials
            && x.iter()
                .zip(&self.probabilities)
                .all(|(&k, &p)| k == 0 || p > S::ZERO))
    }
    /// Returns the covariance matrix `n (diag(p) − p pᵀ)`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the `k × k` result cannot be stored.
    /// - `NumericalInstability` if an entry is outside the range of `S`.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        let d = self.categories();
        self.options.admit::<S>(d.checked_mul(d).ok_or_else(|| {
            invalid(scioxide_core::argument!("self"), "covariance size overflow")
        })?)?;
        let mut out = Matrix::zeros(d, d)?;
        for i in 0..d {
            for j in 0..d {
                let pi = self.probabilities[i].into_f64();
                let pj = self.probabilities[j].into_f64();
                out.view_mut().set(
                    i,
                    j,
                    S::try_from_f64_lossy(
                        self.trials as f64 * pi * (if i == j { 1.0 - pi } else { -pj }),
                    )?,
                )?;
            }
        }
        Ok(out)
    }
}
impl<S: RealScalar> LogMass<S, &[usize]> for Multinomial<S> {
    fn log_pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        if !self.contains_support(x)? {
            return Ok(S::NEG_INFINITY);
        }
        if self.trials == 0 {
            return Ok(S::ZERO);
        }
        // Independent Poisson masses conditioned on their sum give the
        // multinomial. Loader's deviances avoid large log-factorial differences.
        let log = x
            .iter()
            .zip(&self.masses)
            .map(|(&k, mass)| mass.log_mass(k as f64))
            .sum::<f64>()
            - self.log_normalizer;
        if log == f64::NEG_INFINITY {
            Ok(S::NEG_INFINITY)
        } else {
            finite(S::try_from_f64_lossy(log)?)
        }
    }
}
impl<S: RealScalar> Mass<S, &[usize]> for Multinomial<S> {
    fn pmf(&self, x: &[usize]) -> Result<S, SciOxideError> {
        Ok(self.log_pmf(x)?.exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for Multinomial<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        super::converted(
            self.probabilities
                .iter()
                .map(|p| self.trials as f64 * p.into_f64()),
        )
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for Multinomial<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        super::converted(self.probabilities.iter().map(|p| {
            let p = p.into_f64();
            self.trials as f64 * p * (1.0 - p)
        }))
    }
}
impl<S: RealScalar> RowLaw<S, usize> for Multinomial<S> {
    fn width(&self) -> usize {
        self.categories()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.parameter_bytes()
    }
    // Conditional binomial draws, with one word allowance for the whole vector.
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [usize],
        _: Scratch<'_, usize>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        // Prepared complementary fractions retain tiny later categories.
        let mut n = self.trials;
        let mut left = self.options.work_budget().get();
        out.fill(0);
        let last = out.len() - 1;
        for (value, &(p, reflected)) in out[..last].iter_mut().zip(&self.conditional) {
            if n == 0 {
                break;
            }
            let draw = super::super::binomial::conditional_binomial(n, p, &mut words, &mut left)?;
            *value = if reflected { n - draw } else { draw };
            n -= *value;
        }
        out[out.len() - 1] = n;
        Ok(())
    }
}
vector_sampling!(Multinomial, usize, 0);
