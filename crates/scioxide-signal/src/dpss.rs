//! Discrete prolate spheroidal sequences (Slepian tapers).
use crate::util::window_finite as finite;
use crate::util::{empty, invalid};
use crate::windows::{compensated, count, extended_len, normalize};
use crate::{WindowNormalization, WindowSymmetry};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Complex, ExecutionContext, NormSelection, RealScalar, ResidualReport, ResidualScaling,
    ResidualScalingRule, SciOxideError, SolveCounts,
};
use scioxide_fft::{
    FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec, Normalization,
};
use scioxide_linalg::{
    EigenIndexRange, EigenvalueOrder, HermitianEigenDecomposition, HermitianEigenOptions,
    LinalgWorkspace,
};

/// Sampling, scaling and eigensolver options of DPSS tapers.
///
/// The tapers are computed in `f64` for every output type, so the eigensolver
/// tolerance and iteration budget are in `f64` units. The defaults are
/// symmetric sampling, unit-energy tapers and the default
/// [`HermitianEigenOptions`].
///
/// ```
/// use scioxide_signal::{DpssOptions, WindowNormalization, WindowSymmetry};
/// let options = DpssOptions::new()
///     .with_symmetry(WindowSymmetry::Periodic)
///     .with_normalization(WindowNormalization::UnitPeak);
/// assert_eq!(options.symmetry(), WindowSymmetry::Periodic);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DpssOptions {
    symmetry: WindowSymmetry,
    normalization: WindowNormalization,
    eigen: HermitianEigenOptions<f64>,
}
impl Default for DpssOptions {
    fn default() -> Self {
        Self {
            symmetry: WindowSymmetry::Symmetric,
            normalization: WindowNormalization::UnitEnergy,
            eigen: HermitianEigenOptions::default(),
        }
    }
}
impl DpssOptions {
    /// Creates the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the sampling convention. Periodic tapers are computed one sample
    /// longer and then drop their last sample.
    pub const fn with_symmetry(mut self, symmetry: WindowSymmetry) -> Self {
        self.symmetry = symmetry;
        self
    }
    /// Sets the scaling of each returned taper.
    pub const fn with_normalization(mut self, normalization: WindowNormalization) -> Self {
        self.normalization = normalization;
        self
    }
    /// Sets the options of the tridiagonal eigensolver, including its
    /// tolerance and its iteration budget for bisection and inverse iteration.
    /// Construction sets the index range and ordering itself, to select the
    /// largest `taper_count` eigenpairs in descending order.
    pub const fn with_eigen(mut self, eigen: HermitianEigenOptions<f64>) -> Self {
        self.eigen = eigen;
        self
    }
    /// Returns the sampling convention.
    pub const fn symmetry(self) -> WindowSymmetry {
        self.symmetry
    }
    /// Returns the taper scaling.
    pub const fn normalization(self) -> WindowNormalization {
        self.normalization
    }
    /// Returns the eigensolver options.
    pub const fn eigen(self) -> HermitianEigenOptions<f64> {
        self.eigen
    }
}

/// Termination status of a DPSS computation.
///
/// A computation that fails returns an error instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DpssTermination {
    /// The eigenpairs and all requested tapers were computed. This makes no
    /// claim about their accuracy; see [`DpssReport::residual`].
    Completed,
}

/// Work counts and eigenpair residual of a DPSS computation.
#[derive(Clone, Copy, Debug)]
pub struct DpssReport {
    counts: SolveCounts,
    residual: ResidualReport<f64>,
}
impl DpssReport {
    /// Returns the termination status, which is always
    /// [`DpssTermination::Completed`].
    pub const fn termination(self) -> DpssTermination {
        DpssTermination::Completed
    }
    /// Returns the eigensolver work counts, including nested levels; see
    /// [Solve accounting](scioxide_core#solve-accounting).
    pub const fn counts(self) -> SolveCounts {
        self.counts
    }
    /// Returns the largest scaled eigenpair residual over the computed tapers.
    ///
    /// For an eigenpair `(λ, v)` of the commuting tridiagonal matrix `T`
    /// divided by `N²` (see [`DpssTapers`]), the scaled residual is
    ///
    /// ```text
    /// ‖(T − λI) v‖_∞ / ((‖T‖_∞ + |λ|) ‖v‖_∞),
    /// ```
    ///
    /// with the denominator at least `f64::MIN_POSITIVE`. It is measured
    /// before periodic truncation and normalization.
    pub const fn residual(self) -> ResidualReport<f64> {
        self.residual
    }
}

/// Discrete prolate spheroidal sequences (DPSS, Slepian tapers): holds the
/// sampled tapers and their concentrations.
///
/// For the sampled length `N` (the requested length, plus one for
/// [`WindowSymmetry::Periodic`]) and the half-bandwidth `W = NW/N` in cycles
/// per sample, the taper `v_k` of order `k` maximizes the spectral
/// concentration
///
/// ```text
/// λ_k = ∫_{−W}^{W} |V_k(f)|² df / ∫_{−1/2}^{1/2} |V_k(f)|² df
/// ```
///
/// among sequences orthogonal to `v_0, …, v_{k−1}`, where `V_k` is the DTFT of
/// `v_k`. The tapers are the eigenvectors of the symmetric tridiagonal matrix
/// that commutes with the concentration problem,
///
/// ```text
/// T[i, i] = ((N − 1)/2 − i)² cos(2πW),   T[i − 1, i] = T[i, i − 1] = i (N − i)/2,
/// ```
///
/// ordered by decreasing eigenvalue. Construction selects the largest `K`
/// eigenpairs by bisection and inverse iteration with `O(N K)` storage, and
/// computes each `λ_k` from the autocorrelation of `v_k` with an FFT. Tapers of even order
/// have a positive sum, and the first significant sample of a taper of odd
/// order is positive.
/// A taper of length one is `[1]` with concentration `2W`.
///
/// The tapers are stored as consecutive rows. All arithmetic is in `f64`.
///
/// # References
///
/// D. Slepian, "Prolate spheroidal wave functions, Fourier analysis, and
/// uncertainty — V: The discrete case," *Bell Syst. Tech. J.* 57(5),
/// 1371–1430, 1978.
#[derive(Clone, Debug)]
pub struct DpssTapers<S: RealScalar> {
    length: usize,
    tapers: Vec<S>,
    ratios: Vec<f64>,
    report: DpssReport,
}
impl<S: RealScalar> DpssTapers<S> {
    /// Computes `taper_count` DPSS tapers of length `length` with the
    /// time-half-bandwidth product `time_bandwidth = NW`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `time_bandwidth` is NaN or infinite.
    /// - `InvalidInput` (`Empty`) if `length` is zero.
    /// - `InvalidInput` if `taper_count` lies outside `1..=length`, or if
    ///   `time_bandwidth` lies outside `(0, length/2)`.
    /// - `ConvergenceFailure` or `NumericalInstability` if the tridiagonal
    ///   eigensolver fails.
    /// - `NumericalInstability` (`LossOfSignificance`) if a computed
    ///   concentration lies outside `[0, 1]` by more than `64 ε N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{DpssOptions, DpssTapers, DpssTermination};
    /// // Three tapers of length 16 with NW = 2.5.
    /// let tapers = DpssTapers::<f64>::new(16, 2.5, 3, DpssOptions::default())?;
    /// assert_eq!(tapers.report().termination(), DpssTermination::Completed);
    /// assert!(tapers.report().residual().scaled_norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        length: usize,
        time_bandwidth: f64,
        taper_count: usize,
        options: DpssOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            length,
            time_bandwidth,
            taper_count,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Computes the tapers like [`new`](Self::new), with the execution context
    /// `cx`.
    ///
    /// The eigensolver and the FFTs check cancellation and the deadline of
    /// `cx`. The eigensolver runs sequentially for every execution policy.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` on
    /// cancellation or deadline expiry.
    pub fn build(
        length: usize,
        time_bandwidth: f64,
        taper_count: usize,
        options: DpssOptions,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let count = |n| count(n, scioxide_core::argument!("length"));
        crate::util::check_real_slice_finite(&[time_bandwidth])?;
        if length == 0 {
            return Err(empty(
                scioxide_core::argument!("length"),
                "DPSS requires a nonempty length",
            ));
        }
        if taper_count == 0 || taper_count > length {
            return Err(invalid(
                scioxide_core::argument!("taper_count"),
                "DPSS requires 1 <= tapers <= length",
            ));
        }
        if time_bandwidth <= 0.0 || time_bandwidth >= count(length)? / 2.0 {
            return Err(invalid(
                scioxide_core::argument!("time_bandwidth"),
                "DPSS requires 0 < NW < length/2",
            ));
        }
        let size = length
            .checked_mul(taper_count)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("dpss_tapers")))?;
        let extended = extended_len(length, options.symmetry)?;
        let n = count(extended)?;
        let bandwidth = time_bandwidth / n;
        cx.check()?;
        if extended == 1 {
            return Self::single(bandwidth, options);
        }
        let mut diagonal = try_with_capacity(extended, scioxide_core::quantity!("dpss_diagonal"))?;
        let mut off_diagonal =
            try_with_capacity(extended - 1, scioxide_core::quantity!("dpss_off_diagonal"))?;
        // Divide the commuting matrix by N^2: this preserves eigenvectors and
        // keeps all coefficients bounded independently of length.
        for i in 0..extended {
            let x = count(i)? / n;
            diagonal.push(
                ((1.0 - 1.0 / n) / 2.0 - x).powi(2)
                    * (2.0 * core::f64::consts::PI * bandwidth).cos(),
            );
            if i > 0 {
                off_diagonal.push(x * (1.0 - x) / 2.0);
            }
        }
        let mut eigen = HermitianEigenDecomposition::empty();
        let eigen_report = HermitianEigenDecomposition::tridiagonal_into(
            &diagonal,
            &off_diagonal,
            &options
                .eigen
                .with_index_range(EigenIndexRange::new(extended - taper_count, extended)?)
                .with_ordering(EigenvalueOrder::Descending),
            &mut eigen,
            &mut LinalgWorkspace::new(),
            cx,
        )?;
        let matrix_norm = diagonal
            .iter()
            .enumerate()
            .map(|(i, d)| {
                d.abs()
                    + if i > 0 { off_diagonal[i - 1].abs() } else { 0. }
                    + off_diagonal.get(i).copied().unwrap_or(0.).abs()
            })
            .fold(0.0_f64, f64::max);
        let mut residual = ResidualReport::unscaled(0., NormSelection::Infinity)?;
        let mut tapers = try_with_capacity(size, scioxide_core::quantity!("dpss_tapers"))?;
        let mut ratios =
            try_with_capacity(taper_count, scioxide_core::quantity!("dpss_concentration"))?;
        let mut taper = try_filled(
            extended,
            0.0,
            scioxide_core::quantity!("dpss_taper_preparation"),
        )?;
        let mut kernel = try_filled(
            extended,
            0.0,
            scioxide_core::quantity!("dpss_concentration_kernel"),
        )?;
        kernel[0] = 2.0 * bandwidth;
        for (lag, value) in kernel.iter_mut().enumerate().skip(1) {
            let lag = count(lag)?;
            *value = (2.0 * core::f64::consts::PI * bandwidth * lag).sin()
                / (core::f64::consts::PI * lag);
        }
        let mut concentration = Concentration::new(extended, cx)?;
        for k in 0..taper_count {
            cx.check()?;
            for (i, value) in taper.iter_mut().enumerate() {
                *value = eigen.eigenvectors().view().get(i, k)?;
            }
            let eigenvalue = eigen.eigenvalues()[k];
            let raw = taper
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let left = if i > 0 {
                        off_diagonal[i - 1] * taper[i - 1]
                    } else {
                        0.
                    };
                    let right = if i + 1 < extended {
                        off_diagonal[i] * taper[i + 1]
                    } else {
                        0.
                    };
                    compensated([left, (diagonal[i] - eigenvalue) * v, right].into_iter()).abs()
                })
                .fold(0.0_f64, f64::max);
            let denominator = (matrix_norm + eigenvalue.abs())
                * taper.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let candidate = ResidualReport::new(
                raw,
                NormSelection::Infinity,
                ResidualScaling::new(
                    ResidualScalingRule::Reference,
                    denominator.max(f64::MIN_POSITIVE),
                )?,
            )?;
            if candidate.scaled_norm() >= residual.scaled_norm() {
                residual = candidate;
            }
            orient(&mut taper, k);
            ratios.push(concentration.ratio(&taper, &kernel, cx)?);
            normalize(
                &mut taper[..length],
                options.normalization,
                scioxide_core::argument!("options.normalization"),
            )?;
            for &value in &taper[..length] {
                tapers.push(S::try_from_f64_lossy(value)?);
            }
        }
        Ok(Self {
            length,
            tapers,
            ratios,
            report: DpssReport {
                counts: eigen_report.counts(),
                residual,
            },
        })
    }

    /// The length-one sequence: the 1×1 eigenproblem has eigenvector `[1]` with
    /// zero residual and no eigensolver work. A unit impulse has a flat spectrum,
    /// so its concentration in `[-W, W]` is exactly `2 W = 2 NW`, the value the
    /// concentration formula gives at length one.
    fn single(bandwidth: f64, options: DpssOptions) -> Result<Self, SciOxideError> {
        let mut taper = [1.0];
        normalize(
            &mut taper,
            options.normalization,
            scioxide_core::argument!("options.normalization"),
        )?;
        let mut tapers = try_with_capacity(1, scioxide_core::quantity!("dpss_tapers"))?;
        tapers.push(S::try_from_f64_lossy(taper[0])?);
        let mut ratios = try_with_capacity(1, scioxide_core::quantity!("dpss_concentration"))?;
        ratios.push(checked_concentration(2.0 * bandwidth, 1)?);
        Ok(Self {
            length: 1,
            tapers,
            ratios,
            report: DpssReport {
                counts: SolveCounts::new(),
                residual: ResidualReport::unscaled(0., NormSelection::Infinity)?,
            },
        })
    }

    /// Returns the number of samples in each taper.
    pub const fn length(&self) -> usize {
        self.length
    }
    /// Returns the number of tapers.
    pub fn taper_count(&self) -> usize {
        self.ratios.len()
    }
    /// Returns all tapers as consecutive rows of [`Self::length`] samples.
    pub fn samples(&self) -> &[S] {
        &self.tapers
    }
    /// Returns the concentration `λ_k` of each taper, the fraction of its
    /// energy in `|f| ≤ W`.
    ///
    /// The ratios are computed for the unit-energy tapers of the sampled
    /// length `N`, before periodic truncation and normalization.
    pub fn concentration_ratios(&self) -> &[f64] {
        &self.ratios
    }
    /// Returns the eigensolver work counts and residual.
    pub const fn report(&self) -> DpssReport {
        self.report
    }
    /// Returns the taper of order `index`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index` is not less than
    /// [`Self::taper_count`].
    pub fn taper(&self, index: usize) -> Result<&[S], SciOxideError> {
        if index >= self.taper_count() {
            return Err(invalid(
                scioxide_core::argument!("index"),
                "DPSS taper index out of range",
            ));
        }
        Ok(&self.tapers[index * self.length..(index + 1) * self.length])
    }
}
fn orient(taper: &mut [f64], order: usize) {
    let direction = if order.is_multiple_of(2) {
        compensated(taper.iter().copied())
    } else {
        let peak = taper.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        taper
            .iter()
            .copied()
            .find(|x| x.abs() > peak * 1.0e-7)
            .unwrap_or(0.0)
    };
    if direction < 0.0 {
        taper.iter_mut().for_each(|x| *x = -*x);
    }
}
/// Reused zero-padded real transforms for linear autocorrelation.
struct Concentration {
    forward: FftPlan<f64>,
    inverse: FftPlan<f64>,
    data: Vec<f64>,
    bins: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}
impl Concentration {
    fn new(length: usize, context: ExecutionContext<'_>) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("dpss_autocorrelation");
        let fft_len = length
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| SciOxideError::size_overflow(quantity))?;
        let spec = |direction| {
            FftPlanSpec::new(fft_len, direction, Normalization::Inverse, FftLayout::Real)
        };
        let mut cache =
            scioxide_fft::FftPlanCache::new(scioxide_fft::FftCachePolicy::new(1, usize::MAX)?)?;
        let forward = cache.plan(spec(FftDirection::Forward), context)?;
        let inverse = cache.plan(spec(FftDirection::Inverse), context)?;
        let fft_scratch = forward
            .workspace_requirements(())?
            .complex_len()
            .max(inverse.workspace_requirements(())?.complex_len());
        Ok(Self {
            forward,
            inverse,
            data: try_filled(fft_len, 0.0, quantity)?,
            bins: try_filled(fft_len / 2 + 1, Complex::new(0.0, 0.0), quantity)?,
            scratch: try_filled(fft_scratch, Complex::new(0.0, 0.0), quantity)?,
        })
    }

    fn ratio(
        &mut self,
        taper: &[f64],
        kernel: &[f64],
        context: ExecutionContext<'_>,
    ) -> Result<f64, SciOxideError> {
        context.check()?;
        let energy = compensated(taper.iter().map(|x| x * x));
        self.data.fill(0.0);
        self.data[..taper.len()].copy_from_slice(taper);
        let logical_len = self.data.len();
        self.forward.execute_into(
            &self.data,
            FftOutput::hermitian(logical_len, &mut self.bins),
            &mut self.scratch,
            context,
        )?;
        self.bins
            .iter_mut()
            .for_each(|z| *z = Complex::new(z.norm_sqr(), 0.0));
        self.inverse.execute_into(
            FftInput::hermitian(logical_len, &self.bins),
            &mut self.data,
            &mut self.scratch,
            context,
        )?;
        let terms = compensated(
            kernel
                .iter()
                .zip(&self.data)
                .skip(1)
                .map(|(&weight, &correlation)| 2.0 * weight * correlation),
        );
        checked_concentration(kernel[0] + terms / energy, taper.len())
    }
}

fn checked_concentration(ratio: f64, length: usize) -> Result<f64, SciOxideError> {
    let ratio = finite(ratio)?;
    let slack = 64.0 * f64::EPSILON * count(length, scioxide_core::argument!("length"))?;
    if ratio < -slack || ratio > 1.0 + slack {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "DPSS concentration outside rounding envelope",
        ));
    }
    Ok(ratio.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_concentration_matches_direct_linear_autocorrelation() -> Result<(), SciOxideError> {
        let context = ExecutionContext::sequential_native();
        for n in [2, 3, 17, 65] {
            let taper: Vec<f64> = (0..n).map(|i| (0.7 * i as f64).cos()).collect();
            let kernel: Vec<f64> = (0..n)
                .map(|lag| {
                    if lag == 0 {
                        0.25
                    } else {
                        (core::f64::consts::PI * 0.25 * lag as f64).sin()
                            / (core::f64::consts::PI * lag as f64)
                    }
                })
                .collect();
            let energy = compensated(taper.iter().map(|x| x * x));
            let reference = kernel[0]
                + compensated((1..n).map(|lag| {
                    2.0 * kernel[lag]
                        * compensated(
                            taper[..n - lag]
                                .iter()
                                .zip(&taper[lag..])
                                .map(|(a, b)| a * b),
                        )
                })) / energy;
            let mut work = Concentration::new(n, context)?;
            for _ in 0..2 {
                let actual = work.ratio(&taper, &kernel, context)?;
                assert!((actual - reference).abs() < 32.0 * f64::EPSILON);
            }
        }
        assert!(matches!(Concentration::new(usize::MAX, context), Err(error)
            if error.kind() == scioxide_core::SciOxideErrorKind::SizeOverflow));
        Ok(())
    }

    #[test]
    fn concentration_guard_classifies_numerical_failure_and_only_clamps_roundoff() {
        assert!(matches!(checked_concentration(1.1, 1), Err(e)
            if matches!(e.view(), scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::LossOfSignificance, .. })));
        assert_eq!(checked_concentration(1. + f64::EPSILON, 1).ok(), Some(1.));
        assert_eq!(checked_concentration(-f64::EPSILON, 1).ok(), Some(0.));
    }
}
