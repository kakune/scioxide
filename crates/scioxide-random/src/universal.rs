use crate::law::invalid;
use crate::{Distribution, InversionOptions, LogCdf, LogDensity, LogSurvival, Rng, Support};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::{Budget, ExtendedInterval, RealScalar, SciOxideError};

// Generated scalar `Distribution` batches share the persistent storage
// estimate supplied by `$bytes`, which counts against the sampler's byte limit.
macro_rules! scalar_batches {
    ($receiver:ident, $options:expr, $bytes:expr) => {
        fn sample_n_into<R: Rng<S>>(
            &$receiver,
            rng: &mut R,
            policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
            output: &mut [S],
            workspace: &mut crate::SamplingWorkspace<S>,
        ) -> Result<(), SciOxideError> {
            crate::workspace::sample_scalars(
                rng,
                policy,
                output,
                S::ZERO,
                workspace,
                $options.max_bytes.saturating_sub($bytes),
                |rng| $receiver.sample(rng),
            )
        }

        fn sample_n<R: Rng<S>>(&$receiver, rng: &mut R, n: usize) -> Result<Vec<S>, SciOxideError> {
            crate::workspace::admit_bytes(
                scioxide_core::backend::checked_add(
                    crate::workspace::storage_bytes::<S>(n)?,
                    $bytes,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                $options.max_bytes,
            )?;
            crate::workspace::sample_owned(rng, n, S::ZERO, |rng| $receiver.sample(rng))
        }
    };
}

// Generated indexed batch form shared by the universal samplers.
macro_rules! indexed_method {
    ($receiver:ident, $options:expr, $bytes:expr) => {
        /// Fills `output` with independent samples, sample `i` drawn from the stream
        /// `root.split(i)`.
        ///
        /// `root` does not advance. The samples do not depend on the options' execution policy,
        /// which only schedules them. The output is written only if every sample succeeds.
        ///
        /// # Errors
        ///
        /// - The errors of a single draw, of `root.split`, and of `cx` (cancellation).
        /// - `ResourceLimit` if the staging buffer would exceed `max_bytes`.
        /// - `AllocationFailure` if the staging buffer cannot be allocated.
        pub fn sample_indexed_into<R: Rng<S> + crate::SplittableRng + Send + Sync>(
            &$receiver,
            root: &R,
            output: &mut [S],
            workspace: &mut crate::SamplingWorkspace<S>,
            cx: scioxide_core::ExecutionContext<'_>,
        ) -> Result<(), SciOxideError>
        where
            P: Sync,
        {
            indexed(
                root,
                output,
                workspace,
                cx,
                $options,
                $bytes,
                0,
                || Ok(()),
                |rng, _| $receiver.sample(rng),
            )
        }
    };
}

mod adaptive;
pub use adaptive::{
    AdaptiveRejectionSampler, AdaptiveSamplingWorkspace, TransformedDensitySampler,
};

/// Transformation `T` of a density `f` whose concavity a [`ConcaveDensity`] guarantees.
///
/// A density is `T`-concave if `T(f(x))` is concave on its support (Hörmann 1995).
/// [`Log`](Self::Log) gives the log-concave densities.
/// [`NegativeInverseSqrt`](Self::NegativeInverseSqrt) gives a larger class that includes, for
/// example, Student's t with `ν ≥ 1` and the Cauchy density. See [`AdaptiveRejectionSampler`]
/// for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityTransform {
    /// `T(f) = ln f`: log-concave densities.
    Log,
    /// `T(f) = −1/√f`: `T_{−1/2}`-concave densities.
    NegativeInverseSqrt,
}

/// Continuous density on the real line that you supply to a universal sampler, possibly
/// unnormalized.
///
/// The implementor guarantees that the density has positive finite total mass and an interval
/// as support. [`LogDensity::log_pdf`] must be finite in the interior of the support, may be
/// `−∞` at an endpoint where the density is zero, and must return the same value for the same
/// argument. The samplers rely on these properties but can check them only at the points they
/// evaluate.
///
/// See [`RatioUniformsSampler`] for an example.
pub trait UserDensity<S: RealScalar>: LogDensity<S> + Support<S> {
    /// Checks the parameters of the density and confirms the guarantees of this trait.
    ///
    /// # Errors
    ///
    /// Return an error if the guarantees do not hold or cannot be established. The samplers
    /// return it unchanged.
    fn validate_density(&self) -> Result<(), SciOxideError>;
}
impl<S: RealScalar, P: UserDensity<S> + ?Sized> UserDensity<S> for &P {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        (**self).validate_density()
    }
}

/// Differentiable [`UserDensity`] that is `T`-concave for a [`DensityTransform`] `T`.
///
/// [`AdaptiveRejectionSampler`] and [`TransformedDensitySampler`] build their envelopes from
/// tangents, so the concavity must hold on the whole support. See [`AdaptiveRejectionSampler`]
/// for an example.
pub trait ConcaveDensity<S: RealScalar>: UserDensity<S> {
    /// Returns `(ln f)′(x)` at a point `x` in the interior of the support.
    ///
    /// # Errors
    ///
    /// Return an error if the derivative cannot be evaluated. The samplers return it
    /// unchanged.
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError>;
    /// Confirms that `transform(f)` is concave on the whole support.
    ///
    /// # Errors
    ///
    /// Return an error if the concavity is false or not known. Do not guess.
    fn validate_concavity(&self, transform: DensityTransform) -> Result<(), SciOxideError>;
}
impl<S: RealScalar, P: ConcaveDensity<S> + ?Sized> ConcaveDensity<S> for &P {
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError> {
        (**self).log_density_derivative(x)
    }
    fn validate_concavity(&self, t: DensityTransform) -> Result<(), SciOxideError> {
        (**self).validate_concavity(t)
    }
}

/// Rectangle `(0, u_max] × [v_min, v_max]` that encloses the ratio-of-uniforms region of a
/// density.
///
/// For a shift `c`, the region is `{(u, v) : 0 < u ≤ √f(c + v/u)}`. It lies in the rectangle
/// when
///
/// ```text
/// u_max ≥ sup √f(x),    v_min ≤ inf (x − c) √f(x),    v_max ≥ sup (x − c) √f(x),
/// ```
///
/// with the extrema over the whole support. See [`RatioUniformsSampler`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct RatioUniformsBounds<S: RealScalar> {
    /// Shift `c` in `x = c + v/u`. A value near the mode gives a smaller rectangle.
    pub shift: S,
    /// Upper bound `u_max > 0` of `√f(x)`.
    pub u_max: S,
    /// Lower bound `v_min ≤ 0` of `(x − c) √f(x)`.
    pub v_min: S,
    /// Upper bound `v_max ≥ 0` of `(x − c) √f(x)`.
    pub v_max: S,
}
/// [`UserDensity`] with a known enclosing rectangle for [`RatioUniformsSampler`].
///
/// See [`RatioUniformsSampler`] for an example.
pub trait BoundedDensity<S: RealScalar>: UserDensity<S> {
    /// Returns a rectangle that encloses the ratio-of-uniforms region, tails included.
    ///
    /// # Errors
    ///
    /// Return an error if finite enclosing bounds are not known.
    fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<S>, SciOxideError>;
}
impl<S: RealScalar, P: BoundedDensity<S> + ?Sized> BoundedDensity<S> for &P {
    fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<S>, SciOxideError> {
        (**self).ratio_uniforms_bounds()
    }
}
/// Work and storage limits, and indexed-batch scheduling, for the universal samplers.
///
/// The universal samplers are [`AdaptiveRejectionSampler`], [`TransformedDensitySampler`],
/// [`RatioUniformsSampler`] and [`InversionSampler`]. Exhausting an evaluation or generator-call
/// limit is `ConvergenceFailure`, and exceeding `max_bytes` is `ResourceLimit`. Every rejection
/// loop also stops after 1,024 proposals per sample (see [work limits](crate#work-limits)). The
/// samplers check the options when they are created.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_random::UniversalSamplingOptions;
/// let options = UniversalSamplingOptions::default()
///     .with_sample_evaluations(Budget::try_from(256)?)
///     .with_table_nodes(32);
/// assert_eq!(options.table_nodes(), 32);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct UniversalSamplingOptions {
    preparation_evaluations: Budget,
    sample_evaluations: Budget,
    rng_calls: Budget,
    table_nodes: usize,
    max_bytes: usize,
    execution_policy: scioxide_core::ExecutionPolicy,
}
impl Default for UniversalSamplingOptions {
    fn default() -> Self {
        Self {
            preparation_evaluations: Budget::ONE.saturating_add(65_535),
            sample_evaluations: Budget::ONE.saturating_add(4095),
            rng_calls: Budget::ONE.saturating_add(65_535),
            table_nodes: 128,
            max_bytes: 64 * 1024 * 1024,
            execution_policy: scioxide_core::ExecutionPolicy::Sequential,
        }
    }
}
impl UniversalSamplingOptions {
    /// Returns the maximum number of density and derivative evaluations while a sampler is
    /// created (default 65,536).
    pub const fn preparation_evaluations(self) -> Budget {
        self.preparation_evaluations
    }
    /// Returns the options with preparation evaluation limit `value`.
    pub const fn with_preparation_evaluations(mut self, value: Budget) -> Self {
        self.preparation_evaluations = value;
        self
    }
    /// Returns the maximum number of density, derivative and tail evaluations per sample,
    /// including table refinement (default 4,096).
    pub const fn sample_evaluations(self) -> Budget {
        self.sample_evaluations
    }
    /// Returns the options with per-sample evaluation limit `value`.
    pub const fn with_sample_evaluations(mut self, value: Budget) -> Self {
        self.sample_evaluations = value;
        self
    }
    /// Returns the maximum number of uniform draws per sample, including redraws of zero
    /// (default 65,536).
    pub const fn rng_calls(self) -> Budget {
        self.rng_calls
    }
    /// Returns the options with uniform-draw limit `value`.
    pub const fn with_rng_calls(mut self, value: Budget) -> Self {
        self.rng_calls = value;
        self
    }
    /// Returns the maximum number of tangent points of an adaptive envelope (default 128, at
    /// least 2).
    pub const fn table_nodes(self) -> usize {
        self.table_nodes
    }
    /// Returns the options with at most `value` tangent points.
    pub const fn with_table_nodes(mut self, value: usize) -> Self {
        self.table_nodes = value;
        self
    }
    /// Returns the maximum storage in bytes per operation, counting the sampler and the
    /// staging buffer (default 64 MiB).
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
    /// Returns the options with byte limit `value`.
    pub const fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns how `sample_indexed_into` schedules samples (default sequential). It does not
    /// change the samples.
    pub const fn execution_policy(self) -> scioxide_core::ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the options with execution policy `value`.
    pub const fn with_execution_policy(mut self, value: scioxide_core::ExecutionPolicy) -> Self {
        self.execution_policy = value;
        self
    }
}
/// Work done by one successful sample or sampler creation.
///
/// See [`RatioUniformsSampler`] for an example.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SamplingDiagnostics {
    /// Number of proposals, accepted or rejected.
    pub proposals: usize,
    /// Number of rejected proposals.
    pub rejections: usize,
    /// Number of density, derivative or tail evaluations.
    pub evaluations: usize,
    /// Number of uniform draws, including redraws of zero.
    pub rng_calls: usize,
}
/// Sample from a universal sampler, with the work it took.
///
/// See [`RatioUniformsSampler`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct UniversalSample<S: RealScalar> {
    /// The sample, finite and in the support.
    pub value: S,
    /// Work done to produce this sample.
    pub diagnostics: SamplingDiagnostics,
}
impl SamplingDiagnostics {
    /// The shared counts of this work: proposals as iterations, rejections,
    /// and callback (`Function`) and RNG (`Sample`) provider calls.
    fn counts(self) -> scioxide_core::EvaluationCounts {
        scioxide_core::EvaluationCounts::from_iterations(self.proposals)
            .with_proposals(self.proposals, 0, self.rejections)
            .with_provider_count(scioxide_core::EvaluationRole::Function, self.evaluations)
            .with_provider_count(scioxide_core::EvaluationRole::Sample, self.rng_calls)
    }
}
// The one work ledger of a universal operation: limits are checked before
// work, and exhaustion reports the ledger through the shared constructor.
struct Work {
    options: UniversalSamplingOptions,
    diagnostics: SamplingDiagnostics,
    evaluation_limit: usize,
}
impl Work {
    fn sampling(options: UniversalSamplingOptions) -> Self {
        Self {
            options,
            diagnostics: SamplingDiagnostics::default(),
            evaluation_limit: options.sample_evaluations.get(),
        }
    }
    fn preparing(options: UniversalSamplingOptions) -> Self {
        Self {
            evaluation_limit: options.preparation_evaluations.get(),
            ..Self::sampling(options)
        }
    }
    fn evaluation(&mut self) -> Result<(), SciOxideError> {
        if self.diagnostics.evaluations >= self.evaluation_limit {
            return Err(self.exhausted());
        }
        self.diagnostics.evaluations += 1;
        Ok(())
    }
    fn proposal(&mut self) -> Result<(), SciOxideError> {
        if self.diagnostics.proposals >= crate::util::MAX_SAMPLING_ATTEMPTS {
            return Err(self.exhausted());
        }
        self.diagnostics.proposals += 1;
        Ok(())
    }
    fn uniform<S: RealScalar, R: Rng<S>>(
        &mut self,
        rng: &mut R,
        argument: scioxide_core::Argument,
    ) -> Result<f64, SciOxideError> {
        for _ in 0..crate::util::MAX_SAMPLING_ATTEMPTS {
            if self.diagnostics.rng_calls >= self.options.rng_calls.get() {
                return Err(self.exhausted());
            }
            self.diagnostics.rng_calls += 1;
            let u = crate::util::validated_unit_float(rng, argument)?.into_f64();
            if u > 0.0 {
                return Ok(u);
            }
        }
        Err(self.exhausted())
    }
    fn rejected(&mut self) {
        self.diagnostics.rejections += 1;
    }
    fn exhausted(&self) -> SciOxideError {
        crate::law::exhausted_with(self.diagnostics.counts())
    }
}
fn log_density<S: RealScalar, P: UserDensity<S>>(
    p: &P,
    x: S,
    work: &mut Work,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    work.evaluation()?;
    let log = p.log_pdf(x)?.into_f64();
    if log.is_nan() || log == f64::INFINITY {
        return Err(invalid(argument, "invalid user log density"));
    }
    Ok(log)
}
fn coordinate<S: RealScalar>(
    x: f64,
    argument: scioxide_core::Argument,
) -> Result<S, SciOxideError> {
    if !x.is_finite() {
        return Err(invalid(
            argument,
            "unrepresentable universal sample coordinate",
        ));
    }
    let value = S::try_from_f64_lossy(x)?;
    if !value.is_finite() {
        return Err(invalid(
            argument,
            "unrepresentable universal sample coordinate",
        ));
    }
    Ok(value)
}
fn support<S: RealScalar, P: Support<S>>(
    p: &P,
    argument: scioxide_core::Argument,
) -> Result<(f64, f64), SciOxideError> {
    let (a, b) = p
        .support()?
        .bounds()
        .ok_or_else(|| invalid(argument, "empty universal density support"))?;
    let a = a.map_or(f64::NEG_INFINITY, RealScalar::into_f64);
    let b = b.map_or(f64::INFINITY, RealScalar::into_f64);
    if a >= b {
        return Err(invalid(
            argument,
            "universal density needs nonzero support width",
        ));
    }
    Ok((a, b))
}

/// Sampler for a continuous law you supply, by numerical inversion of its distribution
/// function.
///
/// The law gives `ln F(x)` through [`LogCdf`], `ln(1 − F(x))` through [`LogSurvival`], and an
/// interval support through [`Support`]. Both tails must describe the same normalized law. A
/// sample draws `u` uniform on `(0, 1)` and solves `ln F(x) = ln u` for `u ≤ 1/2`, or
/// `ln(1 − F(x)) = ln(1 − u)` for `u > 1/2`, with
/// [`invert_log_probability_with_options`](crate::invert_log_probability_with_options).
/// Working in the smaller tail keeps the relative accuracy of both tails. The density is never
/// evaluated. The root is found to machine precision in `x`, within
/// [`UniversalSamplingOptions::sample_evaluations`] tail evaluations. [`Self::transport`] maps a
/// given `u`, for example a quasi-random coordinate.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExtendedInterval, SciOxideError};
/// use scioxide_random::{
///     Distribution, InversionSampler, LogCdf, LogSurvival, SeedableRng, SplitMix64,
///     Support, UniversalSamplingOptions,
/// };
/// /// Standard exponential law on `[0, inf)`.
/// struct Exponential;
/// impl Support<f64> for Exponential {
///     fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
///         ExtendedInterval::new(Some(0.0), None)
///     }
/// }
/// impl LogCdf<f64> for Exponential {
///     fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> { Ok((-(-x).exp_m1()).ln()) }
/// }
/// impl LogSurvival<f64> for Exponential {
///     fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> { Ok(-x) }
/// }
/// let sampler = InversionSampler::new(Exponential)?;
/// let median = sampler.transport(0.5)?;
/// assert!((median.value - core::f64::consts::LN_2).abs() < 1e-12);
/// assert!(median.diagnostics.evaluations > 0);
/// let x = sampler.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(x >= 0.0);
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InversionSampler<S: RealScalar, P> {
    provider: P,
    inversion: InversionOptions<S>,
    options: UniversalSamplingOptions,
}
impl<S: RealScalar, P: LogCdf<S> + LogSurvival<S> + Support<S>> InversionSampler<S, P> {
    /// Creates the sampler for `provider`.
    ///
    /// It uses the default [`UniversalSamplingOptions`](crate::UniversalSamplingOptions);
    /// [`Self::with_options`] sets the limits.
    ///
    /// # Errors
    ///
    /// - The error of `provider.support()`.
    /// - `InvalidInput` if the support is empty or a single point.
    /// - `ResourceLimit` if the sampler exceeds `max_bytes`.
    pub fn new(provider: P) -> Result<Self, SciOxideError> {
        Self::with_options(provider, UniversalSamplingOptions::default())
    }

    /// Creates the sampler of [`Self::new`] with the limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`UniversalSamplingOptions`](crate::UniversalSamplingOptions).
    pub fn with_options(
        provider: P,
        options: UniversalSamplingOptions,
    ) -> Result<Self, SciOxideError> {
        support(&provider, scioxide_core::argument!("provider"))?;
        crate::workspace::admit_bytes(core::mem::size_of::<Self>(), options.max_bytes)?;
        let limit = options.sample_evaluations;
        let inversion = InversionOptions::default()
            .with_root_options(
                scioxide_rootfind::ScalarRootOptions::default()
                    .with_max_iterations(limit)
                    .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                        scioxide_core::Tolerance::from_machine_epsilon(),
                    )),
            )
            .with_work_budget(limit);
        Ok(Self {
            provider,
            inversion,
            options,
        })
    }
    /// Returns the quantile `F⁻¹(u)` for `0 < u < 1`, without a generator.
    ///
    /// Use it to map quasi-random coordinates to the law. The diagnostics count the tail
    /// evaluations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `u` is not in `(0, 1)`, or the root is not finite or not in the
    ///   support.
    /// - `ConvergenceFailure` if the root solve fails, with its termination reason.
    /// - Errors of the tail functions, unchanged.
    pub fn transport(&self, u: S) -> Result<UniversalSample<S>, SciOxideError> {
        if !u.is_finite() || u <= S::ZERO || u >= S::ONE {
            return Err(invalid(
                scioxide_core::argument!("u"),
                "inversion transport requires 0 < u < 1",
            ));
        }
        let upper = u > S::HALF;
        let log = if upper { (-u).ln_1p() } else { u.ln() };
        let report = crate::invert_log_probability_with_options(
            &self.provider,
            log,
            upper,
            &self.inversion,
        )?;
        let value = crate::inversion::report_value(&report)?;
        if !value.is_finite() || !self.provider.contains(value)? {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "inversion produced invalid sample",
            ));
        }
        Ok(UniversalSample {
            value,
            diagnostics: SamplingDiagnostics {
                evaluations: report.evaluations(),
                ..SamplingDiagnostics::default()
            },
        })
    }
    /// Draws one sample and reports the work it took.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::transport`].
    /// - `ConvergenceFailure` if the uniform-draw limit is exhausted.
    /// - Errors of `rng`.
    pub fn sample_report<R: Rng<S>>(
        &self,
        rng: &mut R,
    ) -> Result<UniversalSample<S>, SciOxideError> {
        let mut work = Work::sampling(self.options);
        work.proposal()?;
        let u = coordinate(
            work.uniform(rng, scioxide_core::argument!("rng"))?,
            scioxide_core::argument!("self"),
        )?;
        let sample = self.transport(u)?;
        work.diagnostics.evaluations = sample.diagnostics.evaluations;
        Ok(UniversalSample {
            value: sample.value,
            diagnostics: work.diagnostics,
        })
    }
}
impl<S: RealScalar, P: LogCdf<S> + LogSurvival<S> + Support<S>> Distribution<S>
    for InversionSampler<S, P>
{
    scalar_batches!(self, self.options, core::mem::size_of::<Self>());
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        Ok(self.sample_report(rng)?.value)
    }
}

/// Sampler for a density you supply, by the ratio-of-uniforms method.
///
/// If `(u, v)` is uniform on the region `{0 < u ≤ √f(c + v/u)}`, then `x = c + v/u` has density
/// proportional to `f` (Kinderman and Monahan 1977). The sampler draws `(u, v)` uniformly from
/// the enclosing rectangle given by [`BoundedDensity::ratio_uniforms_bounds`] and accepts when
/// `2 ln u ≤ ln f(x)`. Each proposal costs two uniforms and at most one density evaluation. The
/// expected number of proposals is the rectangle area `u_max (v_max − v_min)` divided by half
/// the total mass of `f`. The density may be unnormalized.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExtendedInterval, SciOxideError};
/// use scioxide_random::{
///     BoundedDensity, LogDensity, RatioUniformsBounds, RatioUniformsSampler,
///     SeedableRng, SplitMix64, Support, UniversalSamplingOptions, UserDensity,
/// };
/// /// Unnormalized standard normal density `exp(-x^2 / 2)`.
/// struct Gaussian;
/// impl Support<f64> for Gaussian {
///     fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
///         ExtendedInterval::new(None, None)
///     }
/// }
/// impl LogDensity<f64> for Gaussian {
///     fn log_pdf(&self, x: f64) -> Result<f64, SciOxideError> { Ok(-0.5 * x * x) }
/// }
/// impl UserDensity<f64> for Gaussian {
///     fn validate_density(&self) -> Result<(), SciOxideError> { Ok(()) } // mass sqrt(2 pi)
/// }
/// impl BoundedDensity<f64> for Gaussian {
///     // sup sqrt(f) = 1 and sup |x| sqrt(f) = sqrt(2 / e) < 0.86.
///     fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<f64>, SciOxideError> {
///         Ok(RatioUniformsBounds { shift: 0.0, u_max: 1.0, v_min: -0.86, v_max: 0.86 })
///     }
/// }
/// let sampler = RatioUniformsSampler::new(Gaussian)?;
/// let draw = sampler.sample_report(&mut SplitMix64::from_seed(7))?;
/// assert!(draw.value.is_finite());
/// let work = draw.diagnostics;
/// assert!(work.proposals == work.rejections + 1 && work.rng_calls >= 2 * work.proposals);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// A. J. Kinderman and J. F. Monahan, "Computer generation of random variables using the ratio
/// of uniform deviates", *ACM Trans. Math. Softw.* 3(3), 1977, 257–260.
#[derive(Debug, Clone)]
pub struct RatioUniformsSampler<S: RealScalar, P> {
    provider: P,
    bounds: RatioUniformsBounds<S>,
    options: UniversalSamplingOptions,
}
impl<S: RealScalar, P: BoundedDensity<S>> RatioUniformsSampler<S, P> {
    /// Creates the sampler from the bounds that `provider` reports. The bounds are not
    /// searched for or estimated.
    ///
    /// It uses the default [`UniversalSamplingOptions`](crate::UniversalSamplingOptions);
    /// [`Self::with_options`] sets the limits.
    ///
    /// # Errors
    ///
    /// - The errors of `provider.validate_density()`, `provider.support()` and
    ///   `provider.ratio_uniforms_bounds()`.
    /// - `InvalidInput` if the support is empty or a single point, or a bound is not finite,
    ///   `u_max ≤ 0`, `v_min > 0`, `v_max < 0` or `v_min ≥ v_max`.
    /// - `ResourceLimit` if the sampler exceeds `max_bytes`.
    pub fn new(provider: P) -> Result<Self, SciOxideError> {
        Self::with_options(provider, UniversalSamplingOptions::default())
    }

    /// Creates the sampler of [`Self::new`] with the limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`UniversalSamplingOptions`](crate::UniversalSamplingOptions).
    pub fn with_options(
        provider: P,
        options: UniversalSamplingOptions,
    ) -> Result<Self, SciOxideError> {
        provider.validate_density()?;
        support(&provider, scioxide_core::argument!("provider"))?;
        let bounds = provider.ratio_uniforms_bounds()?;
        if ![bounds.shift, bounds.u_max, bounds.v_min, bounds.v_max]
            .iter()
            .all(|v| v.is_finite())
            || bounds.u_max <= S::ZERO
            || bounds.v_min >= bounds.v_max
            || bounds.v_min > S::ZERO
            || bounds.v_max < S::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                "invalid ratio-of-uniforms rectangle",
            ));
        }
        crate::workspace::admit_bytes(core::mem::size_of::<Self>(), options.max_bytes)?;
        Ok(Self {
            provider,
            bounds,
            options,
        })
    }
    /// Draws one sample and reports the work it took.
    ///
    /// Each evaluated point is checked against the rectangle, so wrong bounds are detected when
    /// a proposal reveals them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if an evaluated point lies outside the rectangle, a proposal is not
    ///   finite in `S`, or the log density is NaN or `+∞`.
    /// - `ConvergenceFailure` if 1,024 proposals are rejected or an evaluation or uniform-draw
    ///   limit is exhausted.
    /// - Errors of the density and of `rng`, unchanged.
    pub fn sample_report<R: Rng<S>>(
        &self,
        rng: &mut R,
    ) -> Result<UniversalSample<S>, SciOxideError> {
        let mut work = Work::sampling(self.options);
        let b = self.bounds;
        loop {
            work.proposal()?;
            let unit_u = work.uniform(rng, scioxide_core::argument!("rng"))?;
            let log_u = b.u_max.into_f64().ln() + unit_u.ln();
            let q = work.uniform(rng, scioxide_core::argument!("rng"))?;
            let v = b.v_min.into_f64() * (1.0 - q) + b.v_max.into_f64() * q;
            let x = coordinate::<S>(
                b.shift.into_f64() + (v / b.u_max.into_f64()) / unit_u,
                scioxide_core::argument!("self"),
            )?;
            if self.provider.contains(x)? {
                let log = log_density(
                    &self.provider,
                    x,
                    &mut work,
                    scioxide_core::argument!("self"),
                )?;
                let root = (0.5 * log).exp();
                let transformed = (x.into_f64() - b.shift.into_f64()) * root;
                if root > b.u_max.into_f64()
                    || transformed < b.v_min.into_f64()
                    || transformed > b.v_max.into_f64()
                {
                    return Err(invalid(
                        scioxide_core::argument!("self"),
                        "observed ratio-of-uniforms envelope violation",
                    ));
                }
                if 2.0 * log_u <= log {
                    return Ok(UniversalSample {
                        value: x,
                        diagnostics: work.diagnostics,
                    });
                }
            }
            work.rejected();
        }
    }
}
impl<S: RealScalar, P: BoundedDensity<S>> Distribution<S> for RatioUniformsSampler<S, P> {
    scalar_batches!(self, self.options, core::mem::size_of::<Self>());
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        Ok(self.sample_report(rng)?.value)
    }
}

// Scalar adapter over the one indexed shell: `retained` bytes of the sampler
// and `per_worker` scratch count against the options' byte limit.
#[allow(clippy::too_many_arguments)]
fn indexed<S: RealScalar, R: Rng<S> + crate::SplittableRng + Send + Sync, W: Send>(
    root: &R,
    output: &mut [S],
    workspace: &mut crate::SamplingWorkspace<S>,
    execution: scioxide_core::ExecutionContext<'_>,
    options: UniversalSamplingOptions,
    retained: usize,
    per_worker: usize,
    make: impl FnMut() -> Result<W, SciOxideError>,
    draw: impl Fn(&mut R, &mut W) -> Result<S, SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    crate::workspace::sample_indexed(
        root,
        output,
        1,
        S::ZERO,
        workspace,
        execution,
        options.execution_policy,
        crate::workspace::IndexedBytes {
            limit: options.max_bytes,
            retained,
            per_worker,
        },
        make,
        |rng, scratch, row| {
            row[0] = draw(rng, scratch)?;
            Ok(())
        },
    )
}

impl<S: RealScalar, P: LogCdf<S> + LogSurvival<S> + Support<S>> InversionSampler<S, P> {
    indexed_method!(self, self.options, core::mem::size_of::<Self>());
}
impl<S: RealScalar, P: BoundedDensity<S>> RatioUniformsSampler<S, P> {
    indexed_method!(self, self.options, core::mem::size_of::<Self>());
}

/// Continuous law given by closures for its logarithmic tails, with its support.
///
/// `log_cdf(x)` returns `ln F(x) = ln P(X ≤ x)` and `log_sf(x)` returns `ln P(X > x)`. Both
/// must describe the same normalized law with interval support `support`, and return `−∞` for
/// zero probability. The adapter implements [`LogCdf`], [`LogSurvival`] and [`Support`], so it
/// serves [`InversionSampler`], [`invert_log_probability`](crate::invert_log_probability) and
/// [`log_interval_probability`](crate::log_interval_probability). It forwards each query to one
/// closure call and allocates nothing. The caller gives the guarantees of those traits.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// use scioxide_random::{LogTailsFn, invert_log_probability, log_interval_probability};
/// // Standard exponential law on [0, ∞): ln F(x) = ln(1 − e^(−x)), ln P(X > x) = −x.
/// let law = LogTailsFn::new(
///     |x: f64| Ok((-(-x).exp_m1()).ln()),
///     |x: f64| Ok(-x),
///     ExtendedInterval::new(Some(0.0), None)?,
/// );
/// let median = invert_log_probability(&law, 0.5_f64.ln(), false)?;
/// assert!((median - core::f64::consts::LN_2).abs() < 1e-12);
/// let log_p = log_interval_probability(&law, 1.0, 2.0)?;
/// assert!((log_p.exp() - ((-1.0_f64).exp() - (-2.0_f64).exp())).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LogTailsFn<S: RealScalar, F, G> {
    log_cdf: F,
    log_sf: G,
    support: ExtendedInterval<S>,
}
impl<S, F, G> LogTailsFn<S, F, G>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
    G: Fn(S) -> Result<S, SciOxideError>,
{
    /// Wraps the closures for `ln P(X ≤ x)` and `ln P(X > x)` and the support.
    pub const fn new(log_cdf: F, log_sf: G, support: ExtendedInterval<S>) -> Self {
        Self {
            log_cdf,
            log_sf,
            support,
        }
    }
}
impl<S, F, G> LogCdf<S> for LogTailsFn<S, F, G>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        (self.log_cdf)(x)
    }
}
impl<S, F, G> LogSurvival<S> for LogTailsFn<S, F, G>
where
    S: RealScalar,
    G: Fn(S) -> Result<S, SciOxideError>,
{
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        (self.log_sf)(x)
    }
}
impl<S: RealScalar, F, G> Support<S> for LogTailsFn<S, F, G> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(self.support)
    }
}

/// `T`-concave density given by closures for `ln f` and its derivative, with its support.
///
/// `log_pdf(x)` returns `ln f(x)` of a possibly unnormalized density `f` with interval
/// support `support`, and `derivative(x)` returns `(ln f)′(x)` in the interior. `transform` is
/// the [`DensityTransform`] `T` for which `T(f)` is concave on the whole support. A log-concave
/// density is also `−1/√f`-concave, so a density declared [`DensityTransform::Log`] serves both
/// [`AdaptiveRejectionSampler`] and [`TransformedDensitySampler`], and one declared
/// [`DensityTransform::NegativeInverseSqrt`] serves only the latter. The adapter implements
/// [`Support`], [`LogDensity`], [`UserDensity`] and [`ConcaveDensity`]. It forwards each query
/// to one closure call and allocates nothing. The caller gives the guarantees of
/// [`UserDensity`] and [`ConcaveDensity`], which the samplers check only at the points they
/// evaluate.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// use scioxide_random::{
///     ConcaveDensityFn, DensityTransform, Distribution, SeedableRng, SplitMix64,
///     TransformedDensitySampler,
/// };
/// // Cauchy density 1/(1 + x²): −1/√f = −√(1 + x²) is concave, ln f is not.
/// let density = ConcaveDensityFn::new(
///     |x: f64| Ok(-(x * x).ln_1p()),
///     |x: f64| Ok(-2.0 * x / (1.0 + x * x)),
///     DensityTransform::NegativeInverseSqrt,
///     ExtendedInterval::new(None, None)?,
/// );
/// let sampler = TransformedDensitySampler::new(density, &[-1.0, 0.0, 1.0])?;
/// let mut rng = SplitMix64::from_seed(7);
/// let mut below = 0;
/// for _ in 0..1000 {
///     below += usize::from(sampler.sample(&mut rng)? <= 1.0);
/// }
/// assert!((below as f64 / 1000.0 - 0.75).abs() < 0.05); // P(X ≤ 1) = 3/4
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConcaveDensityFn<S: RealScalar, F, D> {
    log_pdf: F,
    derivative: D,
    transform: DensityTransform,
    support: ExtendedInterval<S>,
}
impl<S, F, D> ConcaveDensityFn<S, F, D>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
    D: Fn(S) -> Result<S, SciOxideError>,
{
    /// Wraps the closures for `ln f(x)` and `(ln f)′(x)`, the concavity class and the support.
    pub const fn new(
        log_pdf: F,
        derivative: D,
        transform: DensityTransform,
        support: ExtendedInterval<S>,
    ) -> Self {
        Self {
            log_pdf,
            derivative,
            transform,
            support,
        }
    }
}
impl<S: RealScalar, F, D> Support<S> for ConcaveDensityFn<S, F, D> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(self.support)
    }
}
impl<S, F, D> LogDensity<S> for ConcaveDensityFn<S, F, D>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        (self.log_pdf)(x)
    }
}
impl<S, F, D> UserDensity<S> for ConcaveDensityFn<S, F, D>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S, F, D> ConcaveDensity<S> for ConcaveDensityFn<S, F, D>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
    D: Fn(S) -> Result<S, SciOxideError>,
{
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError> {
        (self.derivative)(x)
    }
    fn validate_concavity(&self, transform: DensityTransform) -> Result<(), SciOxideError> {
        // A log-concave density is also −1/√f-concave.
        if transform == self.transform || self.transform == DensityTransform::Log {
            Ok(())
        } else {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("transform"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "density is not declared concave under the sampler's transform",
            ))
        }
    }
}

/// Density given by a closure for `ln f`, with an enclosing ratio-of-uniforms rectangle and its
/// support.
///
/// `log_pdf(x)` returns `ln f(x)` of a possibly unnormalized density `f` with interval support
/// `support`, and `bounds` encloses its ratio-of-uniforms region as [`RatioUniformsBounds`]
/// describes. The adapter implements [`Support`], [`LogDensity`], [`UserDensity`] and
/// [`BoundedDensity`], so it serves [`RatioUniformsSampler`]. It forwards each query to one
/// closure call and allocates nothing. The caller gives the guarantees of [`UserDensity`] and
/// [`BoundedDensity`], which the sampler checks only at the points it evaluates.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// use scioxide_random::{
///     BoundedDensityFn, Distribution, RatioUniformsBounds, RatioUniformsSampler, SeedableRng,
///     SplitMix64,
/// };
/// // Unnormalized normal density exp(−x²/2): sup √f = 1 and sup |x| √f = √(2/e) < 0.86.
/// let bounds = RatioUniformsBounds { shift: 0.0, u_max: 1.0, v_min: -0.86, v_max: 0.86 };
/// let line = ExtendedInterval::new(None, None)?;
/// let density = BoundedDensityFn::new(|x: f64| Ok(-0.5 * x * x), bounds, line);
/// let sampler = RatioUniformsSampler::new(density)?;
/// let mut rng = SplitMix64::from_seed(7);
/// let mut sum = 0.0;
/// for _ in 0..1000 {
///     sum += sampler.sample(&mut rng)?;
/// }
/// assert!((sum / 1000.0).abs() < 0.15); // mean 0, standard error 0.03
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BoundedDensityFn<S: RealScalar, F> {
    log_pdf: F,
    bounds: RatioUniformsBounds<S>,
    support: ExtendedInterval<S>,
}
impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> BoundedDensityFn<S, F> {
    /// Wraps the closure for `ln f(x)`, the enclosing rectangle and the support.
    pub const fn new(
        log_pdf: F,
        bounds: RatioUniformsBounds<S>,
        support: ExtendedInterval<S>,
    ) -> Self {
        Self {
            log_pdf,
            bounds,
            support,
        }
    }
}
impl<S: RealScalar, F> Support<S> for BoundedDensityFn<S, F> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(self.support)
    }
}
impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> LogDensity<S> for BoundedDensityFn<S, F> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        (self.log_pdf)(x)
    }
}
impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> UserDensity<S>
    for BoundedDensityFn<S, F>
{
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> BoundedDensity<S>
    for BoundedDensityFn<S, F>
{
    fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<S>, SciOxideError> {
        Ok(self.bounds)
    }
}
