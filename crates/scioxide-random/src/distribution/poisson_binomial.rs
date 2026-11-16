//! Poisson binomial distribution: sums of independent Bernoulli variables.
use super::count_math::{self as math};
use super::count_math::{Kernel, count_capabilities};
use super::finite_table::{FiniteTable, storage};
use crate::distribution::LawOptions;
use crate::{BitGenerator, CountSupport};
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SciOxideError};

/// Poisson binomial distribution: the number of successes in independent Bernoulli trials with
/// success probabilities `p₁, …, p_n`.
///
/// The mean is `Σ pᵢ` and the variance `Σ pᵢ (1 − pᵢ)`. Trials with `pᵢ = 1`
/// shift the support and trials with `pᵢ = 0` are dropped. Let `m` be the
/// number of probabilities strictly between zero and one. The constructor
/// builds the mass table by dynamic programming in log space, in O(m²) time
/// and O(m) storage. Mass, distribution and survival queries then take O(1)
/// time, quantiles O(log m) and draws O(m).
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Mean, PoissonBinomial, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// let law = PoissonBinomial::new(&[0.2_f64, 0.5, 0.9])?;
/// assert!((law.pmf(0)? - 0.04).abs() < 1e-14 && (law.pmf(3)? - 0.09).abs() < 1e-14);
/// assert!((law.cdf(1)? - 0.45).abs() < 1e-14 && (law.sf(1)? - 0.55).abs() < 1e-14);
/// assert!((law.mean()? - 1.6).abs() < 1e-14);
/// assert_eq!(law.quantile(0.5)?, 2);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? <= 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PoissonBinomial<S: RealScalar> {
    probabilities: Vec<f64>,
    table: FiniteTable,
    offset: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> PoissonBinomial<S> {
    /// Creates a Poisson binomial law from the success probabilities.
    ///
    /// The law copies the probabilities and stores at most `4m + 3` `f64`
    /// values.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a probability is NaN or infinite.
    /// - `DomainError` if a probability is outside `[0, 1]`.
    /// - `ConvergenceFailure` if the `m(m + 3)/2` table updates exceed the work
    ///   limit.
    /// - `NotImplemented` if the update count overflows `usize`.
    /// - `ResourceLimit` if a table exceeds the configured byte limit.
    /// - `AllocationFailure` if the tables cannot be allocated.
    pub fn new(probabilities: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(probabilities, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        probabilities: &[S],
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let (offset, n) = probabilities
            .iter()
            .try_fold((0usize, 0usize), |(offset, n), &p| {
                let p =
                    math::probability(p, true, true, scioxide_core::argument!("probabilities"))?;
                Ok::<_, SciOxideError>((
                    offset + usize::from(p == 1.0),
                    n + usize::from(p > 0.0 && p < 1.0),
                ))
            })?;
        // m(m+3)/2 updates includes the zero coefficient at each stage.
        let work = n
            .checked_add(3)
            .and_then(|v| n.checked_mul(v))
            .map(|v| v / 2)
            .ok_or_else(|| {
                SciOxideError::not_implemented("Poisson-binomial preparation size overflow")
            })?;
        if work > options.work_budget().get() {
            return Err(math::exhausted());
        }
        options.admit::<f64>(n + 1)?;
        let mut ps = storage(n)?;
        ps.extend(
            probabilities
                .iter()
                .map(|p| p.into_f64())
                .filter(|&p| p > 0.0 && p < 1.0),
        );
        let mut masses = storage(n + 1)?;
        masses.resize(n + 1, f64::NEG_INFINITY);
        masses[0] = 0.0;
        // Dynamic programming over trials in log space:
        // m_j <- m_j (1 - p) + m_{j-1} p. Iterating j downward updates in
        // place without a second buffer. Degenerate p in {0, 1} were removed
        // above (ones only shift the support by `offset`), so every log is
        // finite.
        for (i, &p) in ps.iter().enumerate() {
            for j in (1..=i + 1).rev() {
                masses[j] = crate::law::logadd(masses[j] + (-p).ln_1p(), masses[j - 1] + p.ln())?;
            }
            masses[0] += (-p).ln_1p();
        }
        let table = FiniteTable::new(offset, masses)?;
        Ok(Self {
            probabilities: ps,
            table,
            offset,
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for PoissonBinomial<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (self.offset, Some(self.offset + self.probabilities.len()))
    }
}
impl<S: RealScalar> Kernel<S> for PoissonBinomial<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        Ok(self.table.mass(k))
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        Ok(self.table.tail(k, u))
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(
            super::count_math::analytic_count(self.offset)?
                + self.probabilities.iter().sum::<f64>(),
        )
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        Ok(self.probabilities.iter().map(|p| p * (1.0 - p)).sum())
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        self.probabilities
            .iter()
            .try_fold(self.offset, |count, &p| {
                Ok(count + usize::from(super::discrete_math::bernoulli(p, rng)?))
            })
    }
}
count_capabilities!(PoissonBinomial);
