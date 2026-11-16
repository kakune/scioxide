//! Randomized quasi-Monte Carlo estimates from independent replicates.
use super::plan::invalid;
use super::{QmcFamily, QmcPlan, QmcRandomization};
use crate::{Philox4x64, QuasiRng};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::RangeAccumulator;
use scioxide_core::{
    EvaluationRole, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError, SolveBudgets,
    SolveScope,
};

/// Randomized quasi-Monte Carlo estimate of a mean `∫_[0,1)ᵈ f(x) dx` with a standard error.
///
/// The plan holds `R` independent randomizations of one base sequence. Sobol and Niederreiter
/// base 2 use a linear matrix scramble with digital shift (Matoušek 1998), and Halton uses a
/// random shift modulo 1 (Cranley and Patterson 1976). Each replicate `r` averages `f` over its
/// first `N` points to give `m_r`, and [`Self::estimate`] returns
///
/// ```text
/// m̄ = (1/R) Σᵣ m_r,     SE = √( Σᵣ (m_r − m̄)² / (R (R − 1)) ).
/// ```
///
/// Under independent uniform randomizations, each `m_r` is an unbiased estimate of the integral,
/// apart from finite-grid discretization. Thus `SE` is an empirical standard error of
/// `m̄`, not a bound. The plan stores the `R` randomized copies of the base plan and their
/// O(R d) randomization words, so estimates of several functions reuse them. An estimate costs
/// `R · N` evaluations of `f`. [`Self::estimate_scoped`] runs replicates in parallel under the
/// options' [`ExecutionPolicy`] when its scope has an executor, and the estimate does not depend
/// on the schedule.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan};
/// let root = Philox4x64::from_key([42, 17]);
/// let options = QmcReplicateOptions::default().with_points_per_replicate(256);
/// let plan = QmcReplicatePlan::new(&QmcPlan::sobol(2)?, &root, options)?;
/// // Integral of x * y over the unit square is 1/4.
/// let estimate = plan.estimate(|x: &[f64]| Ok(x[0] * x[1]))?;
/// assert_eq!(estimate.replicate_means.len(), 8);
/// assert!((estimate.mean - 0.25).abs() < 1e-3 && estimate.standard_error < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. L'Ecuyer and C. Lemieux, "Recent advances in randomized quasi-Monte Carlo methods", in
/// *Modeling Uncertainty*, Kluwer, 2002, 419–474.
#[derive(Debug, Clone)]
pub struct QmcReplicatePlan {
    plans: Vec<QmcPlan>,
    points: usize,
    policy: ExecutionPolicy,
}
/// Replicate count, points per replicate and scheduling of a [`QmcReplicatePlan`].
///
/// [`QmcReplicatePlan::new`] checks the values.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_random::QmcReplicateOptions;
/// let options = QmcReplicateOptions::default().with_replicates(16);
/// assert_eq!((options.replicates(), options.points_per_replicate()), (16, 1_024));
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QmcReplicateOptions {
    replicates: usize,
    points_per_replicate: usize,
    execution_policy: ExecutionPolicy,
}
impl Default for QmcReplicateOptions {
    fn default() -> Self {
        Self {
            replicates: 8,
            points_per_replicate: 1_024,
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}
impl QmcReplicateOptions {
    /// Returns the number `R` of independent replicates (default 8, at least 2).
    pub const fn replicates(self) -> usize {
        self.replicates
    }
    /// Returns the options with `value` replicates.
    pub const fn with_replicates(mut self, value: usize) -> Self {
        self.replicates = value;
        self
    }
    /// Returns the number `N` of points per replicate (default 1,024).
    ///
    /// `N` must be positive, and a power of two for Sobol and Niederreiter base 2.
    pub const fn points_per_replicate(self) -> usize {
        self.points_per_replicate
    }
    /// Returns the options with `value` points per replicate.
    pub const fn with_points_per_replicate(mut self, value: usize) -> Self {
        self.points_per_replicate = value;
        self
    }
    /// Returns how replicates are scheduled (default sequential). Estimates do not depend on it.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the options with the scheduling `value`.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution_policy = value;
        self
    }
}
/// Result of [`QmcReplicatePlan::estimate`] and [`QmcReplicatePlan::estimate_scoped`]: the
/// mean of the replicate means and its standard error.
///
/// The standard error is an empirical estimate, not a bound. See [`QmcReplicatePlan`] for an
/// example.
#[derive(Debug, Clone)]
pub struct QmcReplicateEstimate {
    /// Mean `m̄` of the replicate means, summed in replicate order.
    pub mean: f64,
    /// Standard error `SE` of `m̄`: the sample standard deviation of the replicate means divided
    /// by `√R`.
    pub standard_error: f64,
    /// Replicate means `m_r`, in replicate order.
    pub replicate_means: Vec<f64>,
    /// Number `N` of points evaluated per replicate.
    pub points_per_replicate: usize,
}
impl QmcReplicatePlan {
    /// Creates `R` independently randomized copies of `base`.
    ///
    /// Any randomization of `base` is replaced. Replicate `r` draws its randomization from the
    /// stream `root.fork(0x51)` followed by forks on the four big-endian bytes of `r`, so it does
    /// not depend on `R` and `root` does not advance. Preparation stores `R` copies of the plan
    /// and O(R d) randomization words.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `R < 2`, `R − 1` exceeds `u32::MAX`, the point count is zero, the
    ///   point count is not a power of two for Sobol or Niederreiter base 2, or `base` is reverse
    ///   Halton.
    /// - `InvalidInput` if the tree path of `root` is too deep to fork five more times.
    /// - `AllocationFailure` if the plans cannot be allocated.
    pub fn new(
        base: &QmcPlan,
        root: &Philox4x64,
        options: QmcReplicateOptions,
    ) -> Result<Self, SciOxideError> {
        let (replicates, points) = (options.replicates, options.points_per_replicate);
        if replicates < 2 || u32::try_from(replicates - 1).is_err() || points == 0 {
            return Err(invalid(
                scioxide_core::argument!("replicates"),
                "at least two replicates and positive point count required",
            ));
        }
        let mode = match base.family() {
            QmcFamily::Sobol | QmcFamily::NiederreiterBase2 => {
                // Complete 2^m blocks retain digital-net balance. Random shifts
                // give unbiased estimates independently of this balance condition.
                if !points.is_power_of_two() {
                    return Err(invalid(
                        scioxide_core::argument!("points"),
                        "digital replicates require a power of two",
                    ));
                }
                QmcRandomization::LinearMatrixShift
            }
            QmcFamily::Halton => QmcRandomization::RandomShift,
            _ => {
                return Err(invalid(
                    scioxide_core::argument!("base"),
                    "reverse Halton randomized replicates are unsupported",
                ));
            }
        };
        // Each replicate's randomization comes from its own counter-stream
        // path (domain label, then the replicate index as big-endian bytes),
        // so replicate i is identical regardless of how many replicates are
        // requested or in which order they are built. Clone the validated plan
        // directly; checkpoint encoding is unrelated to in-memory preparation.
        let domain = root.fork(0x51)?;
        let mut plans = scioxide_core::backend::try_with_capacity(
            replicates,
            scioxide_core::quantity!("qmc_replicate_plans"),
        )?;
        for index in 0..replicates {
            let mut stream = domain.clone();
            for label in (index as u32).to_be_bytes() {
                stream = stream.fork(label)?;
            }
            let plan = base.clone();
            plans.push(plan.with_randomization(mode, &mut stream)?);
        }
        Ok(Self {
            plans,
            points,
            policy: options.execution_policy,
        })
    }
    /// Returns the randomized plans, in replicate order.
    pub fn plans(&self) -> &[QmcPlan] {
        &self.plans
    }
    /// Returns the number `N` of points per replicate.
    pub const fn points_per_replicate(&self) -> usize {
        self.points
    }
    /// Returns how [`Self::estimate_scoped`] schedules replicates.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Estimates the mean of `function` over `[0, 1)ᵈ` from the replicates.
    ///
    /// Each replicate evaluates `function` at its first `N` points with output type `S` and sums
    /// the values in `f64`. The function runs on the calling thread and allocates O(R d) storage
    /// before `function` is first called. It equals [`Self::estimate_scoped`] with a sequential
    /// context.
    ///
    /// # Errors
    ///
    /// - Any error returned by `function`, unchanged.
    /// - `NonFinite` if `function` returns NaN or an infinity.
    /// - `NumericalInstability` if a final mean or standard error cannot be represented.
    /// - `SizeOverflow` if `R · N` overflows `usize`.
    /// - `AllocationFailure` if storage cannot be allocated.
    ///
    /// No estimate is returned after an error, even if some replicates finished.
    pub fn estimate<S: RealScalar, F: Fn(&[S]) -> Result<S, SciOxideError> + Sync>(
        &self,
        function: F,
    ) -> Result<QmcReplicateEstimate, SciOxideError> {
        self.estimate_scoped(function, ExecutionContext::sequential_native())
    }
    /// Estimates the mean of `function` as [`Self::estimate`] does, as one epoch of `scope`.
    ///
    /// `scope` is an [`ExecutionContext`] for a top-level call or the `&mut SolveContext` of an
    /// enclosing solve. Before `function` is first called, the epoch admits all `R · N` calls as
    /// `Function` evaluations, which also count in every enclosing solve; if an enclosing
    /// evaluation limit does not allow them, `function` is never called. Replicates run on the
    /// executor of the scope under [`Self::execution_policy`], and the estimate does not depend
    /// on the schedule. The call observes the cancellation and deadline of the scope before each
    /// evaluation.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::estimate`].
    /// - `ConvergenceFailure` if an enclosing evaluation limit does not allow `R · N` calls.
    /// - `ExecutionStopped` if the scope is cancelled or past its deadline, and executor
    ///   errors.
    /// - `AccountingInvariant` if the counts of an enclosing solve are inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets};
    /// use scioxide_core::{SolveIdentity, SolveScope, SolveState, algorithm_id};
    /// use scioxide_random::{Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan};
    /// let options = QmcReplicateOptions::default().with_points_per_replicate(256);
    /// let plan = QmcReplicatePlan::new(&QmcPlan::sobol(2)?, &Philox4x64::from_key([4, 2]), options)?;
    /// let id = SolveIdentity::new(ProblemToken::new(1), algorithm_id!("app", "outer", 1), 0, 0);
    /// let scope = SolveScope::from(ExecutionContext::sequential());
    /// scope.run(&mut SolveState::new(), id, SolveBudgets::unbounded(), |parent| {
    ///     let estimate = plan.estimate_scoped(|x: &[f64]| Ok(x[0] * x[1]), &mut *parent)?;
    ///     assert!((estimate.mean - 0.25).abs() < 1e-3);
    ///     // The enclosing solve counts the R · N = 8 · 256 evaluations.
    ///     assert_eq!(parent.provider_count(EvaluationRole::Function)?, 2048);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn estimate_scoped<
        'scope,
        'capability,
        S: RealScalar,
        F: Fn(&[S]) -> Result<S, SciOxideError> + Sync,
    >(
        &self,
        function: F,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<QmcReplicateEstimate, SciOxideError> {
        let calls = scioxide_core::backend::checked_mul(
            self.plans.len(),
            self.points,
            scioxide_core::quantity!("replicate_evaluations"),
        )?;
        let mut means = scioxide_core::backend::try_filled(
            self.plans.len(),
            0.0,
            scioxide_core::quantity!("replicate_means"),
        )?;
        let mut points = scioxide_core::backend::try_with_capacity(
            self.plans.len(),
            scioxide_core::quantity!("replicate_work"),
        )?;
        for plan in &self.plans {
            points.push(scioxide_core::backend::try_filled(
                plan.dim(),
                S::ZERO,
                scioxide_core::quantity!("replicate_point"),
            )?);
        }
        crate::law::run_scoped(
            scope,
            scioxide_core::algorithm_id!("random", "QmcReplicateEstimate", 1),
            SolveBudgets::unbounded(),
            |context| {
                if !context
                    .record_providers(EvaluationRole::Function, calls)?
                    .is_granted()
                {
                    return Err(crate::law::exhausted());
                }
                let execution = context.execution();
                self.execute(execution, points, &mut means, |sequence, point| {
                    let mut sum = RangeAccumulator::new();
                    for _ in 0..self.points {
                        execution.check()?;
                        sequence.next_point_into(point)?;
                        let value = function(point)?.into_f64();
                        if !value.is_finite() {
                            return Err(SciOxideError::non_finite(
                                scioxide_core::NonFiniteOrigin::Callback,
                                "QMC integrand returned a non-finite value",
                            ));
                        }
                        sum.add(value)?;
                    }
                    sum.finish_quotient(self.points as f64)
                })
            },
        )?;
        let (mean, standard_error) = replicate_moments(&means)?;
        Ok(QmcReplicateEstimate {
            mean,
            standard_error,
            replicate_means: means,
            points_per_replicate: self.points,
        })
    }
    /// Runs every replicate: each replicate runs `body` on its own sequence
    /// and scratch and writes its mean; ownership slots are prepared before
    /// dispatch, so results do not depend on the schedule.
    pub(crate) fn execute<S: RealScalar, W: Send>(
        &self,
        execution: ExecutionContext<'_>,
        scratch: impl IntoIterator<Item = W>,
        means: &mut [f64],
        body: impl Fn(&mut super::QmcSequence<S>, &mut W) -> Result<f64, SciOxideError> + Sync,
    ) -> Result<(), SciOxideError> {
        let mut slots = scioxide_core::backend::try_with_capacity(
            self.plans.len(),
            scioxide_core::quantity!("replicate_work"),
        )?;
        for ((plan, scratch), mean) in self.plans.iter().zip(scratch).zip(means) {
            slots.push((plan.sequence::<S>()?, scratch, mean));
        }
        let plan = execution.plan(
            self.policy,
            self.plans.len(),
            scioxide_core::PlanLimits {
                work_per_item: NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: NonZeroUsize::MAX,
            },
        );
        let mut partition_slots = scioxide_core::backend::try_with_capacity(
            plan.worker_count(),
            scioxide_core::quantity!("execution_partitions"),
        )?;
        let mut remaining = slots.as_mut_slice();
        for part in plan.partitions() {
            let (entries, tail) = remaining.split_at_mut(part.range().len());
            remaining = tail;
            partition_slots.push(scioxide_core::backend::PartitionSlot::new(entries));
        }
        execution.execute_partitioned(plan, &partition_slots, &|_, entries, _| {
            for (sequence, scratch, mean) in entries.iter_mut() {
                **mean = body(sequence, scratch)?;
            }
            Ok(())
        })
    }
}

/// Mean and standard error `sqrt(sum (m_i - m)^2 / (n (n - 1)))` of
/// independent replicate means. Exponent-separated deviations preserve the
/// range until the final square root and division.
/// # Errors
/// Reports fewer than two means or an unrepresentable dispersion.
pub(crate) fn replicate_moments(means: &[f64]) -> Result<(f64, f64), SciOxideError> {
    if means.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("means"),
            "replicate moments need two replicates",
        ));
    }
    let n = means.len() as f64;
    let mut total = RangeAccumulator::new();
    for &mean in means {
        total.add(mean)?;
    }
    let mean = total.finish_quotient(n)?;
    let mut squares = scioxide_core::ScaledAccumulator::ZERO;
    for &value in means {
        let deviation = scioxide_core::Scaled::from_value(value)?
            .sub(scioxide_core::Scaled::from_value(mean)?)?;
        squares.add_product(deviation, deviation)?;
    }
    let error = squares
        .total()?
        .div(scioxide_core::Scaled::from_value(n * (n - 1.0))?)?
        .sqrt()?
        .round()?;
    Ok((mean, error))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{DirectionNumberSet, QmcScramble, SeedableRng, SerializableState, SobolDirection};

    #[test]
    fn replicate_dispersion_preserves_extreme_differences() {
        let (mean, error) = replicate_moments(&[-f64::MAX, f64::MAX, f64::MAX]).unwrap();
        assert!((mean / f64::MAX - 1.0 / 3.0).abs() < 1e-15);
        assert!((error / f64::MAX - 2.0 / 3.0).abs() < 1e-15);
        let tiny = f64::from_bits(1);
        let means: Vec<_> = (0..100)
            .map(|i| if i % 2 == 0 { 6.0 * tiny } else { -6.0 * tiny })
            .collect();
        assert_eq!(replicate_moments(&means).unwrap().1, tiny);
        for n in [3, 7, 49] {
            assert_eq!(
                replicate_moments(&vec![f64::MAX; n]).unwrap(),
                (f64::MAX, 0.0)
            );
        }
    }

    #[test]
    fn cloned_replicates_preserve_custom_sources_and_ignore_requested_count() {
        let root = Philox4x64::from_seed(42);
        let root_before = root.export_state().unwrap();
        for base in [
            QmcPlan::sobol(2)
                .unwrap()
                .with_scramble(QmcScramble::DigitalShift(17))
                .unwrap(),
            QmcPlan::sobol_with_direction_numbers(
                2,
                DirectionNumberSet::Custom(vec![SobolDirection::new(1, 0, vec![1]).unwrap()]),
            )
            .unwrap(),
            QmcPlan::niederreiter_base2(2).unwrap(),
            QmcPlan::halton_with_bases(vec![5, 11])
                .unwrap()
                .with_scramble(QmcScramble::DigitPermutation(17))
                .unwrap(),
        ] {
            let before = base.sequence::<f64>().unwrap().export_state().unwrap();
            let short = QmcReplicatePlan::new(
                &base,
                &root,
                QmcReplicateOptions::default()
                    .with_replicates(2)
                    .with_points_per_replicate(16),
            )
            .unwrap();
            let long = QmcReplicatePlan::new(
                &base,
                &root,
                QmcReplicateOptions::default()
                    .with_replicates(4)
                    .with_points_per_replicate(16),
            )
            .unwrap();
            for (a, b) in short.plans().iter().zip(long.plans()) {
                assert_eq!(a.family(), base.family());
                assert_eq!(a.direction_words(), base.direction_words());
                assert_eq!(a.bases(), base.bases());
                assert_eq!(
                    a.sequence::<f64>().unwrap().export_state().unwrap(),
                    b.sequence::<f64>().unwrap().export_state().unwrap()
                );
            }
            assert_eq!(
                base.sequence::<f64>().unwrap().export_state().unwrap(),
                before
            );
            assert_eq!(root.export_state().unwrap(), root_before);
        }
    }
}
