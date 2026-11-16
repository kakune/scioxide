//! Transforms of quasi-random points to multivariate normal and multinomial samples.
use super::plan::invalid;
use crate::{
    Multinomial, MultivariateNormal, OutputPolicy, QmcSequence, QuasiRng, SamplingPolicy,
    SamplingRequirements, SamplingWorkspace, StatePolicy,
};
use alloc::vec::Vec;
use scioxide_core::NumericScalar;
use scioxide_core::{Capacity, RealScalar, SciOxideError};

/// Workspace for [`QmcMultivariateNormal`] and [`QmcMultinomial`]: one quasi-random point and
/// the output staging buffer.
///
/// `S` is the coordinate type and `T` the output element type (`S` for normal samples, `usize`
/// for multinomial counts). Create it with the adapter's `workspace()`, which sizes the point
/// for the adapter's input dimension. The staging buffer for all-or-nothing batches is sized by
/// [`SamplingPolicy::workspace_requirements`] for rows of the law's width. It grows on demand, or
/// ahead of time with [`Self::try_reserve`], and is reused by later calls.
///
/// See [`QmcMultivariateNormal`] for an example.
#[derive(Debug, Clone)]
pub struct QmcTransportWorkspace<S, T = S> {
    coordinates: Vec<S>,
    publication: SamplingWorkspace<T>,
}
impl<S: RealScalar, T> QmcTransportWorkspace<S, T> {
    fn new(coordinates: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            coordinates: scioxide_core::backend::try_filled(
                coordinates,
                S::ZERO,
                scioxide_core::quantity!("qmc_transport_coordinates"),
            )?,
            publication: SamplingWorkspace::new(),
        })
    }
    /// Reports whether the staging buffer already has room for `requirements`.
    ///
    /// # Errors
    ///
    /// - The errors of [`SamplingWorkspace::check`].
    pub fn check(&self, requirements: &SamplingRequirements) -> Result<Capacity, SciOxideError> {
        self.publication.check(requirements)
    }
    /// Grows the staging buffer to hold `requirements`.
    ///
    /// # Errors
    ///
    /// - The errors of [`SamplingWorkspace::try_reserve`].
    pub fn try_reserve(
        &mut self,
        requirements: &SamplingRequirements,
    ) -> Result<(), SciOxideError> {
        self.publication.try_reserve(requirements)
    }
}
/// Quasi-random samples of a [`MultivariateNormal`] law by inversion.
///
/// Each sample takes one point `u ∈ (0, 1)ʳ` of a quasi-random sequence, where `r` is the rank
/// of the covariance. It maps each coordinate through the standard normal quantile,
/// `zₖ = Φ⁻¹(uₖ)`, and returns `x = μ + V Λ^(1/2) z` with the law's factorization. Inversion
/// keeps the low-discrepancy structure coordinate by coordinate, which Box–Muller or rejection
/// would not. A sample costs `r` quantile evaluations and O(d r) arithmetic. A law of rank 0
/// returns `μ` and uses no points.
///
/// Point 0 of an unrandomized Sobol or Niederreiter sequence is the origin, whose quantile is
/// infinite. Skip it, or use a randomized plan. [`Self::sample`] returns one sample,
/// [`Self::sample_n`] `n` samples in a new `Vec`, and [`Self::sample_n_into`] fills a slice you
/// provide under a [`SamplingPolicy`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_random::{MultivariateNormal, QmcMultivariateNormal, QmcPlan, QuasiRng};
/// let covariance = Matrix::from_vec(1, 1, vec![1.0_f64])?;
/// let law = MultivariateNormal::new(&[2.0], &covariance)?;
/// let adapter = QmcMultivariateNormal::new(&law);
/// let plan = QmcPlan::sobol(adapter.input_dimension())?;
/// let mut sequence = plan.sequence::<f64>()?;
/// sequence.skip(1)?; // Exclude the origin, whose normal quantile is infinite.
/// let samples = adapter.sample_n(&mut sequence, 3)?;
/// // Points 1, 2, 3 are 1/2, 3/4, 1/4: the mean and two samples symmetric about it.
/// assert!((samples[0] - 2.0).abs() < 1e-12);
/// assert!((samples[1] + samples[2] - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QmcMultivariateNormal<'a, S: RealScalar + NumericScalar<Real = S>> {
    law: &'a MultivariateNormal<S>,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> QmcMultivariateNormal<'a, S> {
    /// Creates the adapter for `law`, reusing its factorization.
    pub fn new(law: &'a MultivariateNormal<S>) -> Self {
        Self { law }
    }
    /// Returns the dimension the quasi-random sequence must have: the covariance rank `r`.
    ///
    /// It is zero for a point mass.
    pub fn input_dimension(&self) -> usize {
        self.law.rank()
    }
    /// Creates a workspace with room for one point. The staging buffer starts empty.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the point cannot be allocated.
    pub fn workspace(&self) -> Result<QmcTransportWorkspace<S>, SciOxideError> {
        QmcTransportWorkspace::new(self.input_dimension())
    }
    /// Writes `out.len() / d` samples into `out`, one row of `d` values per sample.
    ///
    /// Each sample takes the next point of `sequence`. Laws of rank 0 take no points.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `sequence` does not have `r` coordinates, `work` was made for
    ///   another adapter, or `out.len()` is not a multiple of `d`.
    /// - `DomainError` if a coordinate is 0 or 1, where the normal quantile is infinite.
    /// - `InvalidInput` if `sequence` is exhausted.
    /// - `NumericalInstability` if a sample overflows.
    ///
    /// On error, `out` and `sequence` follow `policy` (see
    /// [batch sampling](crate#batch-sampling)).
    pub fn sample_n_into<'q, P: StatePolicy<QmcSequence<'q, S>>>(
        &self,
        sequence: &mut QmcSequence<'q, S>,
        policy: SamplingPolicy<P>,
        out: &mut [S],
        work: &mut QmcTransportWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let rank = self.law.rank();
        if rank > 0 && sequence.dim() != rank {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(rank),
                scioxide_core::Dimensions::one(sequence.dim()),
            ));
        }
        if work.coordinates.len() != rank {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(rank),
                scioxide_core::Dimensions::one(work.coordinates.len()),
            ));
        }
        crate::workspace::sample_rows(
            sequence,
            policy,
            out,
            self.law.dimension(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            &mut work.publication,
            usize::MAX,
            |sequence, row| {
                // One QMC coordinate per retained eigen-direction (not per
                // output dimension), mapped through the normal quantile.
                // Inversion (rather than Box-Muller or Ziggurat) preserves the
                // low-discrepancy structure coordinate by coordinate.
                if rank > 0 {
                    sequence.next_point_into(&mut work.coordinates)?;
                    for x in &mut work.coordinates {
                        *x = crate::distribution::standard_normal_quantile(*x)?;
                    }
                }
                self.law.transform(&work.coordinates, row)
            },
        )
    }
    /// Returns one sample of `d` values from the next point of `sequence`.
    ///
    /// It equals [`Self::sample_n`] with `n = 1`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::sample_n`].
    pub fn sample<'q>(&self, sequence: &mut QmcSequence<'q, S>) -> Result<Vec<S>, SciOxideError> {
        self.sample_n(sequence, 1)
    }
    /// Returns `n` samples in a new `Vec`, one row of `d` values per sample.
    ///
    /// The samples equal those of [`Self::sample_n_into`]. The output and a workspace are
    /// allocated before the first point. On error nothing is returned and `sequence` stays
    /// advanced past the points it produced.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::sample_n_into`].
    /// - `SizeOverflow` if `n · d` does not fit in `usize`.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn sample_n<'q>(
        &self,
        sequence: &mut QmcSequence<'q, S>,
        n: usize,
    ) -> Result<Vec<S>, SciOxideError> {
        let quantity = scioxide_core::quantity!("gaussian_qmc_output");
        let mut out = scioxide_core::backend::try_filled(
            scioxide_core::backend::checked_mul(n, self.law.dimension(), quantity)?,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            quantity,
        )?;
        let mut work = self.workspace()?;
        let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
        self.sample_n_into(sequence, policy, &mut out, &mut work)?;
        Ok(out)
    }
}
/// Quasi-random samples of a [`Multinomial`] law by inversion, one coordinate per trial.
///
/// For `n` trials and category probabilities `p₁, …, p_K`, the interval `[0, 1)` is split into
/// consecutive pieces of lengths `pₖ / Σⱼ pⱼ`. Each sample takes one point `u ∈ [0, 1)ⁿ` and
/// counts trial `t` in the category whose piece contains `u_t`. The counts sum to `n` exactly
/// and follow the stratification of the point set. A sample costs O(n log K). With zero trials
/// a sample is all zeros and uses no points. [`Self::sample`] returns one count vector,
/// [`Self::sample_n`] `n` of them in a new `Vec`, and [`Self::sample_n_into`] fills a slice you
/// provide under a [`SamplingPolicy`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{Multinomial, QmcMultinomial, QmcPlan};
/// let law = Multinomial::new(2, &[0.25_f64, 0.75])?;
/// let adapter = QmcMultinomial::new(&law)?;
/// let plan = QmcPlan::sobol(adapter.input_dimension())?;
/// let counts = adapter.sample(&mut plan.sequence::<f64>()?)?;
/// assert_eq!(counts, [2, 0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct QmcMultinomial<'a, S: RealScalar> {
    law: &'a Multinomial<S>,
    cumulative: Vec<f64>,
}
impl<'a, S: RealScalar> QmcMultinomial<'a, S> {
    /// Creates the adapter for `law` by computing the category boundaries in `f64`.
    ///
    /// The boundary of the last category with positive probability is exactly 1.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if no category has positive probability, or if a positive probability is
    ///   too small for its piece to have positive length in `f64`.
    /// - `AllocationFailure` if the boundaries cannot be allocated.
    pub fn new(law: &'a Multinomial<S>) -> Result<Self, SciOxideError> {
        let mut cumulative = scioxide_core::backend::try_with_capacity(
            law.categories(),
            scioxide_core::quantity!("multinomial_qmc_intervals"),
        )?;
        let total = law
            .probabilities()
            .iter()
            .map(|p| p.into_f64())
            .sum::<f64>();
        let mut sum = scioxide_core::RangeAccumulator::new();
        let mut previous = 0.0;
        let last = law
            .probabilities()
            .iter()
            .rposition(|p| *p > S::ZERO)
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("law"),
                    "positive category required",
                )
            })?;
        // Right endpoints of the category intervals partitioning [0, 1). The
        // last positive category is pinned to exactly 1 so rounding in the
        // running sum can never leave a gap at the top that would map a
        // coordinate to a zero-probability trailing category.
        for (i, &p) in law.probabilities().iter().enumerate() {
            sum.add(p.into_f64())?;
            let next = if i >= last {
                1.0
            } else {
                sum.clone().finish()? / total
            };
            if p > S::ZERO && next <= previous {
                return Err(invalid(
                    scioxide_core::argument!("law"),
                    "positive QMC interval is not representable",
                ));
            }
            cumulative.push(next);
            previous = next;
        }
        Ok(Self { law, cumulative })
    }
    /// Returns the dimension the quasi-random sequence must have: the number of trials `n`.
    pub fn input_dimension(&self) -> usize {
        self.law.trials()
    }
    /// Creates a workspace with room for one point. The staging buffer starts empty.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the point cannot be allocated.
    pub fn workspace(&self) -> Result<QmcTransportWorkspace<S, usize>, SciOxideError> {
        QmcTransportWorkspace::new(self.input_dimension())
    }
    /// Writes `out.len() / K` count vectors into `out`, one row of `K` counts per sample.
    ///
    /// Each row sums to `n`. Each sample takes the next point of `sequence`, and zero trials take
    /// no points.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `sequence` does not have `n` coordinates, `work` was made for
    ///   another adapter, or `out.len()` is not a multiple of `K`.
    /// - `InvalidInput` if a coordinate equals 1 or `sequence` is exhausted.
    ///
    /// On error, `out` and `sequence` follow `policy` (see
    /// [batch sampling](crate#batch-sampling)).
    pub fn sample_n_into<'q, P: StatePolicy<QmcSequence<'q, S>>>(
        &self,
        sequence: &mut QmcSequence<'q, S>,
        policy: SamplingPolicy<P>,
        out: &mut [usize],
        work: &mut QmcTransportWorkspace<S, usize>,
    ) -> Result<(), SciOxideError> {
        let trials = self.law.trials();
        if trials > 0 && sequence.dim() != trials {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(trials),
                scioxide_core::Dimensions::one(sequence.dim()),
            ));
        }
        if work.coordinates.len() != trials {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(trials),
                scioxide_core::Dimensions::one(work.coordinates.len()),
            ));
        }
        crate::workspace::sample_rows(
            sequence,
            policy,
            out,
            self.law.categories(),
            0,
            &mut work.publication,
            usize::MAX,
            |sequence, row| {
                row.fill(0);
                if trials > 0 {
                    sequence.next_point_into(&mut work.coordinates)?;
                }
                // Each trial consumes one coordinate and is assigned by inverse
                // CDF lookup, so the counts inherit the point set's
                // stratification across trials.
                for &u in &work.coordinates {
                    crate::law::probability(u, scioxide_core::argument!("sequence"))?;
                    if u == S::ONE {
                        return Err(invalid(
                            scioxide_core::argument!("sequence"),
                            "QMC transport requires [0,1)",
                        ));
                    }
                    let category = self.cumulative.partition_point(|&p| p <= u.into_f64());
                    row[category] += 1;
                }
                Ok(())
            },
        )
    }
    /// Returns one count vector of `K` counts from the next point of `sequence`.
    ///
    /// It equals [`Self::sample_n`] with `n = 1`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::sample_n`].
    pub fn sample<'q>(
        &self,
        sequence: &mut QmcSequence<'q, S>,
    ) -> Result<Vec<usize>, SciOxideError> {
        self.sample_n(sequence, 1)
    }
    /// Returns `n` count vectors in a new `Vec`, one row of `K` counts per sample.
    ///
    /// The counts equal those of [`Self::sample_n_into`]. The output and a workspace are
    /// allocated before the first point. On error nothing is returned and `sequence` stays
    /// advanced past the points it produced.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::sample_n_into`].
    /// - `SizeOverflow` if `n · K` does not fit in `usize`.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn sample_n<'q>(
        &self,
        sequence: &mut QmcSequence<'q, S>,
        n: usize,
    ) -> Result<Vec<usize>, SciOxideError> {
        let quantity = scioxide_core::quantity!("multinomial_qmc_output");
        let mut out = scioxide_core::backend::try_filled(
            scioxide_core::backend::checked_mul(n, self.law.categories(), quantity)?,
            0,
            quantity,
        )?;
        let mut work = self.workspace()?;
        let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
        self.sample_n_into(sequence, policy, &mut out, &mut work)?;
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use crate::QmcPlan;
    use scioxide_linalg::Matrix;

    #[test]
    fn gaussian_coordinate_shapes_are_checked_before_consumption() {
        let law = MultivariateNormal::new(&[0.0_f64; 2], &Matrix::identity(2).unwrap()).unwrap();
        let transport = QmcMultivariateNormal::new(&law);
        for (dimensions, scratch) in [(1, 2), (2, 1)] {
            let plan = QmcPlan::sobol(dimensions).unwrap();
            let mut sequence = plan.sequence::<f64>().unwrap();
            let saved = sequence.position();
            let mut out = [123.0; 2];
            let mut work = QmcTransportWorkspace::new(scratch).unwrap();
            crate::error_assertions::dimensions(
                transport
                    .sample_n_into(
                        &mut sequence,
                        SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                        &mut out,
                        &mut work,
                    )
                    .unwrap_err(),
                2,
                1,
            );
            assert_eq!(out, [123.0; 2]);
            assert_eq!(sequence.position(), saved);
        }
    }

    #[test]
    fn multinomial_coordinate_shapes_are_checked_before_consumption() {
        let law = Multinomial::new(2, &[0.5_f64, 0.5]).unwrap();
        let transport = QmcMultinomial::new(&law).unwrap();
        for (dimensions, scratch) in [(1, 2), (2, 1)] {
            let plan = QmcPlan::sobol(dimensions).unwrap();
            let mut sequence = plan.sequence::<f64>().unwrap();
            let saved = sequence.position();
            let mut out = [123; 2];
            let mut work = QmcTransportWorkspace::new(scratch).unwrap();
            crate::error_assertions::dimensions(
                transport
                    .sample_n_into(
                        &mut sequence,
                        SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                        &mut out,
                        &mut work,
                    )
                    .unwrap_err(),
                2,
                1,
            );
            assert_eq!(out, [123; 2]);
            assert_eq!(sequence.position(), saved);
        }
    }
}
