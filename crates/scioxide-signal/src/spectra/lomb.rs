//! Lomb–Scargle periodogram by weighted least-squares sinusoid fits.
use super::*;
use scioxide_core::{NdView, NdViewMut, Tolerance};
/// Normalization of the Lomb–Scargle periodogram. Neither is a density.
///
/// [`LombScargle`] defines `N`, `R₀` and `R(ω)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LombScargleNormalization {
    /// `(N/2)(R₀ − R(ω))`: half the sample count times the reduction of the
    /// weighted mean-square residual, in squared data units.
    Power,
    /// `(R₀ − R(ω)) / R₀`: the fraction of the residual energy of the baseline
    /// that the sinusoid explains, in `[0, 1]`.
    Normalized,
}
/// Baseline fitted together with the sinusoid at each frequency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LombScargleMean {
    /// Fits a constant offset at each frequency. Needs at least three samples
    /// with positive weight.
    Floating,
    /// Assumes a zero baseline. Needs at least two samples with positive
    /// weight.
    Zero,
}
/// Options of a Lomb–Scargle periodogram.
///
/// The defaults are [`LombScargleMean::Floating`],
/// [`LombScargleNormalization::Normalized`] and the rank tolerance
/// [`Tolerance::from_sqrt_epsilon`].
///
/// ```
/// use scioxide_signal::{LombScargleMean, LombScargleNormalization, LombScargleOptions};
///
/// let options = LombScargleOptions::<f64>::default()
///     .with_mean(LombScargleMean::Zero)
///     .with_normalization(LombScargleNormalization::Power);
/// assert_eq!(options.mean(), LombScargleMean::Zero);
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct LombScargleOptions<S: RealScalar> {
    mean: LombScargleMean,
    normalization: LombScargleNormalization,
    tolerance: Tolerance<S>,
}
impl<S: RealScalar> Default for LombScargleOptions<S> {
    fn default() -> Self {
        Self {
            mean: LombScargleMean::Floating,
            normalization: LombScargleNormalization::Normalized,
            tolerance: Tolerance::from_sqrt_epsilon(),
        }
    }
}
impl<S: RealScalar> LombScargleOptions<S> {
    /// Returns the default options listed at [`LombScargleOptions`].
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the baseline fitted with each sinusoid.
    pub fn with_mean(mut self, mean: LombScargleMean) -> Self {
        self.mean = mean;
        self
    }
    /// Sets the normalization of the periodogram.
    pub fn with_normalization(mut self, normalization: LombScargleNormalization) -> Self {
        self.normalization = normalization;
        self
    }
    /// Sets the rank tolerance `(atol, rtol)` of the sinusoid fit.
    ///
    /// At each frequency, the cosine and sine columns are orthogonalized
    /// against the baseline and each other. A column whose weighted norm is
    /// then at most `atol + rtol` contributes no power. Before
    /// orthogonalization each column has weighted norm at most one.
    pub fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns the baseline fitted with each sinusoid.
    pub fn mean(self) -> LombScargleMean {
        self.mean
    }
    /// Returns the normalization of the periodogram.
    pub fn normalization(self) -> LombScargleNormalization {
        self.normalization
    }
    /// Returns the rank tolerance.
    pub fn tolerance(self) -> Tolerance<S> {
        self.tolerance
    }
    fn floating(self) -> bool {
        self.mean == LombScargleMean::Floating
    }
}
/// Computes the Lomb–Scargle periodogram of the samples `input` taken at the
/// times `times`, at the angular frequencies `frequencies`.
///
/// This is [`LombScargle::new`] with unit weights and the default
/// [`LombScargleOptions`], followed by [`LombScargle::power`]. At each
/// frequency `ω` a constant offset and a sinusoid are fitted by least squares
/// ([`LombScargleMean::Floating`]), and the result is the fraction
/// `(R₀ − R(ω)) / R₀` of the baseline residual that the sinusoid explains
/// ([`LombScargleNormalization::Normalized`]): the generalized periodogram of
/// Zechmeister and Kürster. [`LombScargle`] defines `R₀` and `R(ω)` and takes
/// weights and other options. Frequencies are in radians per time unit, and
/// the times need not be sorted. For `N` samples and `M` frequencies the cost
/// is `O(N·M)`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `times` or `frequencies` is empty.
/// - `InvalidInput` (`OutOfRange`) if there are fewer than three samples or a
///   frequency is not positive.
/// - `DimensionMismatch` if `input` and `times` differ in length.
/// - `NonFinite` if a time, frequency or sample is NaN or infinite.
/// - `DomainError` (`Undefined`) if `R₀` is zero: the input is constant.
/// - `NumericalInstability` (`LossOfSignificance`) if a fit explains more than
///   `(1 + 128ε) R₀`, and (`Overflow`) if a phase `ω (tᵢ − t₀)` or a power
///   overflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::lomb_scargle;
/// let times = [0.0, 0.7, 1.9, 2.4, 3.3, 4.1];
/// let input = times.map(|t: f64| 1.5 + (2.0 * t).sin());
/// let power = lomb_scargle(&times, &input, &[1.0, 2.0])?;
/// // An offset sinusoid at ω = 2 is fitted exactly there: normalized power one.
/// assert!((power[1] - 1.0).abs() < 1e-12 && power[0] < 0.9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Zechmeister and M. Kürster, "The generalised Lomb-Scargle periodogram",
/// *Astron. Astrophys.* 496, 577–584, 2009.
pub fn lomb_scargle<S: RealScalar>(
    times: &[S],
    input: &[S],
    frequencies: &[S],
) -> Result<Vec<S>, SciOxideError> {
    LombScargle::new(times, frequencies, None, LombScargleOptions::default())?.power(input)
}
/// Lomb–Scargle periodogram of unevenly sampled real data, for fixed sample
/// times, frequencies and weights.
///
/// For `N` samples `xᵢ` at times `tᵢ` with weights `wᵢ ≥ 0` (all one by
/// default), and an angular frequency `ω`, the periodogram fits
///
/// ```text
/// xᵢ ≈ c + a cos(ω tᵢ) + b sin(ω tᵢ)
/// ```
///
/// by weighted least squares. The offset `c` is fitted for
/// [`LombScargleMean::Floating`] and is zero for [`LombScargleMean::Zero`].
/// With normalized weights `ŵᵢ = wᵢ / Σⱼ wⱼ`, `R₀` is the weighted residual
/// `Σ ŵᵢ rᵢ²` of the baseline alone and `R(ω)` that of the full fit.
/// [`LombScargleNormalization`] turns them into power. With unit weights, a
/// zero baseline and [`LombScargleNormalization::Power`], the result is the
/// classical periodogram of Lomb and Scargle. With a floating mean it is the
/// generalized periodogram of Zechmeister and Kürster.
///
/// Frequencies are angular, in radians per time unit, and the times need not
/// be sorted. Scaling all weights by a positive factor does not change the
/// result. At each frequency, the cosine and sine columns are orthogonalized
/// with reorthogonalization, and a column that is dependent at the rank
/// tolerance of [`LombScargleOptions::with_tolerance`] contributes no power.
/// For `M` frequencies the cost is O(N·M), with one sine and cosine per
/// sample and frequency, and the workspace of one record holds `4N + M`
/// values. [`LombScargle::power_axis_into`] takes every lane along one axis of
/// an array as a record. The value holds the validated times, frequencies and
/// weights and computes nothing at construction.
///
/// # Accuracy
///
/// With `γ` and `η` as under [Accuracy](crate#accuracy), an inner product of
/// `k` terms has absolute error at most `γ(2k + 8) Σ|aᵢbᵢ| + (2k + 8)·η`.
/// Orthogonalization and normalization add ordinary rounding errors, which
/// columns close to the rank cutoff amplify. There is no relative error bound
/// at zero power or where the rank of the fit changes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{LombScargle, LombScargleOptions};
///
/// let times = [0.0, 0.7, 1.9, 2.4, 3.3, 4.1];
/// let frequencies = [1.0, 2.0];
/// let periodogram = LombScargle::new(&times, &frequencies, None, LombScargleOptions::default())?;
/// // A sinusoid at ω = 2 is fitted exactly there: normalized power one.
/// let input = times.map(|t: f64| (2.0 * t).sin());
/// let mut ws = periodogram.workspace(1)?;
/// let mut power = [0.0; 2];
/// let cx = ExecutionContext::sequential_native();
/// periodogram.power_into(&input, &mut power, &mut ws, cx)?;
/// assert!(power[1] > 0.99 && power[0] < power[1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - N. R. Lomb, "Least-squares frequency analysis of unequally spaced data",
///   *Astrophys. Space Sci.* 39, 447–462, 1976.
/// - J. D. Scargle, "Studies in astronomical time series analysis. II.
///   Statistical aspects of spectral analysis of unevenly spaced data",
///   *Astrophys. J.* 263, 835–853, 1982.
/// - M. Zechmeister and M. Kürster, "The generalised Lomb-Scargle periodogram",
///   *Astron. Astrophys.* 496, 577–584, 2009.
#[derive(Clone, Copy, Debug)]
pub struct LombScargle<'a, S: RealScalar> {
    times: &'a [S],
    frequencies: &'a [S],
    weights: Option<&'a [S]>,
    weight_scale: S,
    weight_sum: S,
    options: LombScargleOptions<S>,
}
impl<'a, S: RealScalar> LombScargle<'a, S> {
    /// Prepares the periodogram for sample times `times`, angular frequencies
    /// `frequencies` and optional weights `weights`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a time, frequency or weight is NaN or infinite.
    /// - `InvalidInput` (`Empty`) if `times` or `frequencies` is empty.
    /// - `InvalidInput` (`OutOfRange`) if there are fewer samples, or fewer
    ///   positive weights, than [`LombScargleMean`] needs, a frequency is not
    ///   positive, or a weight is negative.
    /// - `DimensionMismatch` if `weights` and `times` differ in length.
    pub fn new(
        times: &'a [S],
        frequencies: &'a [S],
        weights: Option<&'a [S]>,
        options: LombScargleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        inputs(times)?;
        inputs(frequencies)?;
        // Validate even when no rank calculation has yet occurred.
        options.tolerance.is_residual_ok(S::ZERO, S::ONE)?;
        let minimum = if options.floating() { 3 } else { 2 };
        if times.is_empty() {
            return Err(empty(
                scioxide_core::argument!("times"),
                "at least one sample time is required",
            ));
        }
        if times.len() < minimum {
            return Err(invalid(
                scioxide_core::argument!("times"),
                "too few samples for the selected mean model",
            ));
        }
        if frequencies.is_empty() {
            return Err(empty(
                scioxide_core::argument!("frequencies"),
                "frequencies must be nonempty",
            ));
        }
        if frequencies.iter().any(|&w| w <= S::ZERO) {
            return Err(invalid(
                scioxide_core::argument!("frequencies"),
                "frequencies must be positive",
            ));
        }
        let (weight_scale, weight_sum) = if let Some(w) = weights {
            exact_len(times.len(), w.len())?;
            inputs(w)?;
            if w.iter().any(|&x| x < S::ZERO)
                || w.iter().filter(|&&x| x > S::ZERO).count() < minimum
            {
                return Err(invalid(
                    scioxide_core::argument!("weights"),
                    "weights must be nonnegative with enough positive entries",
                ));
            }
            let scale = w.iter().copied().fold(S::ZERO, S::max);
            (scale, sum_products(w.iter().map(|&x| (x / scale, S::ONE)))?)
        } else {
            (S::ONE, scalar(times.len())?)
        };
        Ok(Self {
            times,
            frequencies,
            weights,
            weight_scale,
            weight_sum,
            options,
        })
    }
    /// Returns the angular frequencies, in radians per time unit, in the order
    /// given.
    pub fn frequencies(&self) -> &[S] {
        self.frequencies
    }
    /// Returns the number of samples `N` that each record must have.
    pub fn sample_len(&self) -> usize {
        self.times.len()
    }
    /// Returns the options.
    pub fn options(&self) -> LombScargleOptions<S> {
        self.options
    }
    /// Returns the workspace size for `lanes` records at once:
    /// `lanes·M + 5N + M` values for `N` samples and `M` frequencies.
    ///
    /// Use one lane for [`LombScargle::power_into`], which needs only `4N + M`
    /// of them, and the number of lanes along the axis for
    /// [`LombScargle::power_axis_into`]: the product of the extents of the
    /// other axes.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        lanes: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("lomb_batch");
        Ok(SignalRequirements::values(checked_add(
            checked_add(
                checked_mul(lanes, self.frequencies.len(), quantity)?,
                self.sample_len(),
                quantity,
            )?,
            checked_add(
                self.frequencies.len(),
                self.kernel_len()?,
                scioxide_core::quantity!("lomb_lane"),
            )?,
            quantity,
        )?))
    }
    /// Allocates a workspace for `lanes` records.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, lanes: usize) -> Result<SignalWorkspace<S>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(lanes)?)
    }
    /// Workspace of one record: the staged output and the kernel scratch.
    fn record_requirements(&self) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(checked_add(
            self.kernel_len()?,
            self.frequencies.len(),
            scioxide_core::quantity!("lomb_work"),
        )?))
    }
    /// Scratch of the per-lane kernel: four sample-length vectors.
    fn kernel_len(&self) -> Result<usize, SciOxideError> {
        checked_mul(
            4,
            self.times.len(),
            scioxide_core::quantity!("lomb_samples"),
        )
    }

    fn root_weights(&self, output: &mut [S]) {
        if let Some(weights) = self.weights {
            for (out, &weight) in output.iter_mut().zip(weights) {
                *out = ((weight / self.weight_scale) / self.weight_sum).sqrt();
            }
        } else {
            output.fill((S::ONE / self.weight_sum).sqrt());
        }
    }
    fn remove_mean(values: &mut [S], weights: &[S]) -> Result<(), SciOxideError> {
        // Reproject to remove the rounding left by the first subtraction.
        for _ in 0..2 {
            let mean = correction(values, weights)?;
            for (x, &weight) in values.iter_mut().zip(weights) {
                *x -= mean * weight;
            }
        }
        Ok(())
    }
    /// Evaluate every frequency of validated `input` into `staged`, using four
    /// sample-length vectors of `work`.
    fn power_staged(
        &self,
        input: &[S],
        staged: &mut [S],
        work: &mut [S],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = input.len();
        let (weights, rest) = work.split_at_mut(n);
        self.root_weights(weights);
        let (y, rest) = rest.split_at_mut(n);
        let (c, rest) = rest.split_at_mut(n);
        let s = &mut rest[..n];
        let amplitude = input.iter().fold(S::ZERO, |a, x| a.max(x.abs()));
        for (i, (out, &x)) in y.iter_mut().zip(input).enumerate() {
            *out = if amplitude == S::ZERO {
                S::ZERO
            } else {
                (x / amplitude) * weights[i]
            };
        }
        if self.options.floating() {
            if input.iter().all(|&x| x == input[0]) {
                y.fill(S::ZERO);
            } else {
                Self::remove_mean(y, weights)?;
            }
        }
        let energy = dot(y, y)?;
        if energy == S::ZERO && self.options.normalization == LombScargleNormalization::Normalized {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("input"),
                scioxide_core::DomainReason::Undefined,
                "normalized power is undefined for zero-energy input",
            ));
        }
        for (&omega, out) in self.frequencies.iter().zip(staged.iter_mut()) {
            context.check()?;
            for (i, (&time, (cosine, sine))) in self
                .times
                .iter()
                .zip(c.iter_mut().zip(s.iter_mut()))
                .enumerate()
            {
                let phase =
                    crate::util::finite(crate::util::finite(time - self.times[0])? * omega)?;
                let (sin, cos) = phase.sin_cos();
                let weight = weights[i];
                *cosine = cos * weight;
                *sine = sin * weight;
            }
            if self.options.floating() {
                Self::remove_mean(c, weights)?;
                Self::remove_mean(s, weights)?;
            }
            // Pivot before rank selection so a vanishing cosine never hides a
            // usable sine. Both columns share the unit weighted design scale.
            if norm(s)? > norm(c)? {
                c.swap_with_slice(s);
            }
            let first = normalize(c, self.options.tolerance)?;
            if first {
                for _ in 0..2 {
                    let projection = correction(c, s)?;
                    for (b, &a) in s.iter_mut().zip(c.iter()) {
                        *b -= projection * a;
                    }
                }
            }
            let second = normalize(s, self.options.tolerance)?;
            let a = if first { dot(c, y)? } else { S::ZERO };
            let b = if second { dot(s, y)? } else { S::ZERO };
            let explained = crate::util::working_sum_products([(a, a), (b, b)].into_iter())?;
            let allowance = scalar::<S>(128)? * S::EPSILON;
            if explained > energy * (S::ONE + allowance) {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::LossOfSignificance,
                    "projection exceeds the reference energy",
                ));
            }
            *out = match self.options.normalization {
                LombScargleNormalization::Normalized => (explained / energy).min(S::ONE),
                LombScargleNormalization::Power => crate::util::checked_factor_product(&[
                    explained.min(energy),
                    amplitude,
                    amplitude,
                    scalar::<S>(n)?,
                    S::HALF,
                ])
                .ok_or_else(crate::util::output_overflow)?,
            };
        }
        Ok(())
    }
    /// Computes the periodogram of `input` at every frequency and writes it to
    /// `output`.
    ///
    /// `input` holds one value per sample time. `output` needs at least one
    /// element per frequency, and `ws` the size of
    /// [`LombScargle::workspace_requirements`] for one lane.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input.len()` differs from the number of
    ///   sample times, or `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `DomainError` (`Undefined`) if the normalization is
    ///   [`LombScargleNormalization::Normalized`] and `R₀` is zero: the input
    ///   is zero, or constant with a floating mean.
    /// - `NumericalInstability` (`LossOfSignificance`) if a fit explains more
    ///   than `(1 + 128ε) R₀`, and (`Overflow`) if a phase `ω (tᵢ − t₀)` or a
    ///   power overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn power_into(
        &self,
        input: &[S],
        output: &mut [S],
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_len(self.times.len(), input.len())?;
        capacity(self.frequencies.len(), output.len())?;
        let buffers = ws.buffers(&self.record_requirements()?)?;
        inputs(input)?;
        cx.check()?;
        let (staged, work) = buffers.values.split_at_mut(self.frequencies.len());
        self.power_staged(input, staged, work, cx)?;
        cx.check()?;
        output[..staged.len()].copy_from_slice(staged);
        Ok(())
    }

    /// Computes the periodogram of `input` at every frequency and returns it.
    ///
    /// # Errors
    ///
    /// - The errors of [`LombScargle::power_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn power(&self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        let mut output = zeros(self.frequencies.len())?;
        let mut ws = SignalWorkspace::reserved(&self.record_requirements()?)?;
        self.power_into(
            input,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes the periodogram of every lane of `input` along axis `axis` at
    /// every frequency and writes it to `output`.
    ///
    /// Each lane of `N` samples, one per sample time, is evaluated as
    /// [`LombScargle::power_into`] evaluates it. `output` has the shape of
    /// `input` with `N` replaced by the number `M` of frequencies. Lanes run in
    /// row-major order of the other axes, and both views may be strided. `ws`
    /// needs the size of [`LombScargle::workspace_requirements`] for the
    /// number of lanes; with it the call does not allocate. The periodograms
    /// are staged in `ws` and written to `output` after every lane succeeds.
    /// The cost is O(N·M) per lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of
    ///   `input`.
    /// - `DimensionMismatch` if the axis length differs from the number of
    ///   sample times, `output` does not have the expected shape, or `ws` is
    ///   too short.
    /// - The errors of [`LombScargle::power_into`] for each lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{LombScargle, LombScargleOptions};
    /// let times = [0.0_f64, 0.7, 1.9, 2.4, 3.3, 4.1];
    /// let periodogram = LombScargle::new(&times, &[1.0, 2.0], None, LombScargleOptions::default())?;
    /// // Two records along axis 1: sinusoids at ω = 1 and ω = 2.
    /// let input = [times.map(f64::sin), times.map(|t| (2.0 * t).sin())].concat();
    /// let mut ws = periodogram.workspace(2)?;
    /// let mut power = [0.0; 4];
    /// let samples = NdView::new(&input, StridedLayout::new(&[2, 6], &[6, 1])?)?;
    /// let output = NdViewMut::new(&mut power, StridedLayout::new(&[2, 2], &[2, 1])?)?;
    /// let cx = ExecutionContext::sequential_native();
    /// periodogram.power_axis_into(samples, 1, output, &mut ws, cx)?;
    /// // Each sinusoid is fitted exactly at its own frequency.
    /// assert!((power[0] - 1.0).abs() < 1e-12 && (power[3] - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn power_axis_into(
        &self,
        input: NdView<'_, '_, S>,
        axis: usize,
        mut output: NdViewMut<'_, '_, S>,
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let shape = input.layout().shape().extents();
        let frequencies = self.frequencies.len();
        let n = self.sample_len();
        let geometry = crate::axis::Geometry::new(shape, axis, n, frequencies)?;
        check_axis_shape(shape, axis, frequencies, output.layout().shape().extents())?;
        let buffers = ws.buffers(&self.workspace_requirements(geometry.batches)?)?;
        cx.check()?;
        let (staged, rest) = buffers.values.split_at_mut(geometry.output_len);
        let (lane, rest) = rest.split_at_mut(n);
        let (result, work) = rest.split_at_mut(frequencies);
        for batch in 0..geometry.batches {
            geometry.gather(&input, lane, batch)?;
            inputs(lane)?;
            cx.check()?;
            self.power_staged(lane, result, work, cx)?;
            geometry.scatter(result, batch, staged)?;
        }
        cx.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }
}
/// Dot product whose sign is checked against an absolute rounding bound.
/// Under cancellation, or when the bound is not proved, the dot product is
/// recomputed exactly.
fn dot<S: RealScalar>(x: &[S], y: &[S]) -> Result<S, SciOxideError> {
    let [value, mass] = scioxide_core::backend::four_lane_sums(x.len().min(y.len()), |i| {
        let product = x[i] * y[i];
        [product, product.abs()]
    });
    let k = scalar::<S>(2 * x.len() + 8)?;
    let ku = k * (S::EPSILON * S::HALF);
    let gamma = ku / (S::ONE - ku);
    // The second denominator encloses the rounded, nonnegative mass reduction.
    let error = (gamma / (S::ONE - gamma) * mass + k * S::MIN_SUBNORMAL).next_after(S::INFINITY);
    if ku < S::HALF && value.is_finite() && error.is_finite() && value.abs() > error {
        Ok(value)
    } else {
        sum_products(x.iter().zip(y).map(|(&a, &b)| (a, b)))
    }
}
/// Native dot for centering and reorthogonalization coefficients. Their
/// second passes cancel by construction, which no sign certificate can accept,
/// and the coefficient only needs the native absolute envelope: the next
/// subtraction removes it to the rounding of the reduced vector. A non-finite
/// candidate replays exact accumulation.
fn correction<S: RealScalar>(x: &[S], y: &[S]) -> Result<S, SciOxideError> {
    let [value] = scioxide_core::backend::four_lane_sums(x.len().min(y.len()), |i| [x[i] * y[i]]);
    if value.is_finite() {
        Ok(value)
    } else {
        sum_products(x.iter().zip(y).map(|(&a, &b)| (a, b)))
    }
}
fn norm<S: RealScalar>(x: &[S]) -> Result<S, SciOxideError> {
    let square = dot(x, x)?;
    if square.is_normal() {
        Ok(square.sqrt())
    } else {
        Ok(x.iter().fold(S::ZERO, |n, &v| n.hypot(v)))
    }
}
fn normalize<S: RealScalar>(x: &mut [S], tolerance: Tolerance<S>) -> Result<bool, SciOxideError> {
    let mut length = norm(x)?;
    let threshold = tolerance.absolute() + tolerance.relative();
    let uncertainty = scalar::<S>(2 * x.len() + 8)? * S::EPSILON * length;
    if (length - threshold).abs() <= uncertainty {
        length = x.iter().fold(S::ZERO, |n, &v| n.hypot(v));
    }
    if tolerance.is_residual_ok(length, S::ONE)? {
        return Ok(false);
    }
    for v in x {
        *v /= length;
    }
    Ok(true)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn certified_dot_replays_cancellation_and_subnormal_products() {
        assert_eq!(dot(&[1e16, 1.0, -1e16, 0.0], &[1.0; 4]).unwrap(), 1.0);
        assert_eq!(dot(&[1e8_f32, 1.0, -1e8, 0.0], &[1.0; 4]).unwrap(), 1.0);
        let tiny32 = f32::from_bits(1);
        assert_eq!(dot(&[tiny32, tiny32], &[0.5, 0.5]).unwrap(), tiny32);
        let tiny = f64::from_bits(1);
        assert_eq!(dot(&[tiny, tiny], &[0.5, 0.5]).unwrap(), tiny);
        assert_eq!(dot(&[0.0; 4], &[1.0; 4]).unwrap(), 0.0);
    }

    #[test]
    fn corrections_stay_native_under_cancellation() {
        // A certified dot replays this cancellation exactly; the correction
        // keeps the native four-lane value inside its absolute envelope.
        let (x, y) = ([1e16, 1.0, -1e16, 0.0], [1.0; 4]);
        // Native lanes round 1e16 + 1 to 1e16, so the native value is zero.
        assert_eq!(correction(&x, &y).unwrap(), 0.0);
        assert_eq!(dot(&x, &y).unwrap(), 1.0);
        let large = [f64::MAX, f64::MAX, -f64::MAX];
        assert_eq!(correction(&large, &[1.0; 3]).unwrap(), f64::MAX);
    }

    #[test]
    fn rank_boundary_and_tiny_norm_use_the_retained_design_scale() {
        let tolerance = Tolerance::new(0.5, 0.0).unwrap();
        let mut boundary = [0.3, 0.4];
        assert!(!normalize(&mut boundary, tolerance).unwrap());
        let mut above = [0.5_f64.next_up(), 0.0];
        assert!(normalize(&mut above, tolerance).unwrap());
        let mut tiny = [1e-300_f64, 0.0];
        assert!(normalize(&mut tiny, Tolerance::new(0.0, 0.0).unwrap()).unwrap());
        assert!((tiny[0] - 1.0).abs() <= f64::EPSILON);
    }
}
