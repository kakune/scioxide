use crate::util::{
    check_real_slice_finite, empty, exact_len, finite, inconsistent, invalid,
    scalar_from_usize_rounded as scalar, unsupported,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Argument, Complex, ExecutionContext, ExecutionPolicy, RealScalar, ResidualTolerance,
    SciOxideError, SolveContext, SolveScope,
};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy, Vector,
};

mod frequency_sampling;
mod least_squares;
mod remez;
mod window_method;

pub use frequency_sampling::design_fir_frequency_sampling;
pub use least_squares::design_fir_least_squares;
pub use remez::{design_fir_remez, design_fir_remez_scoped, design_fir_remez_with_options};
pub use window_method::design_fir_window;

/// Coefficient symmetry of a linear-phase FIR filter.
///
/// For `n` taps `h[0], …, h[n − 1]`, even symmetry means `h[i] = h[n − 1 − i]`
/// and odd symmetry means `h[i] = −h[n − 1 − i]`. With `c = (n − 1)/2` and `f`
/// in multiples of Nyquist, the frequency response is
///
/// ```text
/// H(f) = e^(−iπfc) A(f)      (even symmetry)
/// H(f) = i e^(−iπfc) A(f)    (odd symmetry)
/// ```
///
/// where `A` is the real zero-phase amplitude. Symmetry and the parity of `n`
/// give the four linear-phase types, and some of them force `A` to vanish at a
/// band end:
///
/// | Symmetry | `n` odd | `n` even |
/// |---|---|---|
/// | Even | Type I | Type II, `A(1) = 0` |
/// | Odd | Type III, `A(0) = A(1) = 0` | Type IV, `A(0) = 0` |
///
/// Designs reject a nonzero desired amplitude at a forced zero with
/// `InvalidInput`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirSymmetry {
    /// Equal reflected coefficients (Types I and II).
    Even,
    /// Opposite reflected coefficients (Types III and IV).
    Odd,
}
/// A frequency band with a linear desired amplitude and a weight.
///
/// The band covers the closed interval `[lower, upper]` in multiples of
/// Nyquist (see [Conventions](crate#conventions)). On it the desired zero-phase
/// amplitude `D(f)` varies linearly from `desired_lower` to `desired_upper`.
/// The weight `W` scales the error `A(f) − D(f)` in the minimax criterion of
/// [`design_fir_remez`] and the squared-error integral of
/// [`design_fir_least_squares`]. It defaults to one, and
/// [`design_fir_window`] ignores it.
///
/// Each design checks its bands. It returns `InvalidInput` if the list is
/// empty, if an edge lies outside `[0, 1]`, if `lower ≥ upper`, if a weight is
/// not positive, or if the bands are not in ascending order without overlap.
/// A non-finite edge, amplitude or weight is `NonFinite`.
///
/// ```
/// use scioxide_signal::FirBand;
/// let stop = FirBand::new(0.4_f64, 1.0, 0.0, 0.0).with_weight(8.0);
/// assert_eq!((stop.lower(), stop.upper(), stop.weight()), (0.4, 1.0, 8.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirBand<S: RealScalar> {
    lower: S,
    upper: S,
    desired_lower: S,
    desired_upper: S,
    weight: S,
}
impl<S: RealScalar> FirBand<S> {
    /// Creates the band `[lower, upper]` whose desired zero-phase amplitude
    /// varies linearly from `desired_lower` to `desired_upper`, with unit
    /// weight.
    pub const fn new(lower: S, upper: S, desired_lower: S, desired_upper: S) -> Self {
        Self {
            lower,
            upper,
            desired_lower,
            desired_upper,
            weight: S::ONE,
        }
    }
    /// Sets the relative weight.
    pub const fn with_weight(mut self, weight: S) -> Self {
        self.weight = weight;
        self
    }
    /// Returns the lower edge, in multiples of Nyquist.
    pub const fn lower(self) -> S {
        self.lower
    }
    /// Returns the upper edge, in multiples of Nyquist.
    pub const fn upper(self) -> S {
        self.upper
    }
    /// Returns the desired zero-phase amplitude at the lower edge.
    pub const fn desired_lower(self) -> S {
        self.desired_lower
    }
    /// Returns the desired zero-phase amplitude at the upper edge.
    pub const fn desired_upper(self) -> S {
        self.desired_upper
    }
    /// Returns the relative weight.
    pub const fn weight(self) -> S {
        self.weight
    }
}
/// Gain normalization of a window or frequency-sampling design.
///
/// A normalization frequency outside `[0, 1]` is `InvalidInput`. A designed
/// amplitude of zero at that frequency, including a zero forced by the
/// [`FirSymmetry`], is `DomainError`.
#[derive(Clone, Copy, Debug)]
pub enum FirNormalization<S: RealScalar> {
    /// Keeps the scaling of the designed response.
    None,
    /// Scales the taps so that the zero-phase amplitude `A(frequency)` equals
    /// `gain`.
    At {
        /// Frequency in multiples of Nyquist.
        frequency: S,
        /// Desired signed zero-phase amplitude.
        gain: S,
    },
}
/// Termination status of an FIR design.
///
/// ```
/// use scioxide_signal::FirTermination;
/// assert!(FirTermination::Completed.is_success());
/// assert!(FirTermination::Converged.is_success());
/// assert!(!FirTermination::IterationLimit.is_success());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirTermination {
    /// A direct design (window, frequency sampling or least squares) finished.
    /// It makes no claim about a tolerance.
    Completed,
    /// The Remez exchange met its tolerance on the spread of the alternating
    /// weighted errors.
    Converged,
    /// The Remez exchange reached its iteration limit. The design holds the
    /// last finite approximation.
    IterationLimit,
    /// The Remez exchange stopped because the grid has fewer than `m + 1`
    /// alternating extrema at least as large as the levelled error, where `m`
    /// is the number of independent coefficients. The design holds the last
    /// finite approximation.
    ExchangeStalled,
}
impl FirTermination {
    /// Returns `true` for [`Completed`](Self::Completed) and
    /// [`Converged`](Self::Converged).
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Completed | Self::Converged)
    }
}
/// Taps and diagnostics of a linear-phase FIR design.
///
/// The taps `h[0], …, h[n − 1]` are in convolution order, so filtering computes
/// `y[k] = Σᵢ h[i] x[k − i]`, and they are symmetric according to
/// [`FirSymmetry`]. [`convolve`](crate::convolve)`(input, design.coefficients())`
/// applies the filter to a record and returns the full convolution, and
/// [`to_polyphase`](Self::to_polyphase) prepares repeated or streamed
/// filtering. Direct designs report [`FirTermination::Completed`] with
/// zero iterations. Only [`design_fir_remez`] reports a ripple and extrema,
/// and only [`design_fir_least_squares`] reports a condition estimate. The
/// ripple is measured on the design grid. It estimates the error on the
/// continuous bands but does not bound it.
///
/// ```
/// use scioxide_signal::{FirBand, FirNormalization, FirSymmetry, WindowKind, convolve};
/// let band = FirBand::new(0.0_f64, 0.5, 1.0, 1.0);
/// let (symmetry, normalization) = (FirSymmetry::Even, FirNormalization::None);
/// let design =
///     scioxide_signal::design_fir_window(9, symmetry, &[band], WindowKind::Hann, normalization)?;
/// assert!(design.is_success() && design.iterations() == 0);
/// let taps = design.coefficients();
/// assert!(taps.iter().zip(taps.iter().rev()).all(|(a, b)| a == b));
/// // Filtering a unit impulse returns the taps.
/// let impulse = convolve(&[1.0], taps)?;
/// assert!(impulse.iter().zip(taps).all(|(y, h)| (y - h).abs() < 1e-15));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct FirDesign<S: RealScalar> {
    taps: Vec<S>,
    symmetry: FirSymmetry,
    termination: FirTermination,
    iterations: usize,
    ripple: Option<S>,
    extrema: Vec<S>,
    condition: Option<S>,
}
impl<S: RealScalar> FirDesign<S> {
    /// Returns the taps in convolution order.
    pub fn coefficients(&self) -> &[S] {
        &self.taps
    }
    /// Returns the coefficient symmetry.
    pub fn symmetry(&self) -> FirSymmetry {
        self.symmetry
    }
    /// Returns the termination status.
    pub fn termination(&self) -> FirTermination {
        self.termination
    }
    /// Returns whether the design completed or converged; see
    /// [`FirTermination::is_success`].
    pub fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the number of completed exchange iterations, zero for direct
    /// designs.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the largest weighted error `max |W(f) (A(f) − D(f))|` on the
    /// exchange grid of a Remez design, or `None` for a direct design.
    pub fn weighted_ripple(&self) -> Option<S> {
        self.ripple
    }
    /// Returns the frequencies of the last Remez reference set, in multiples
    /// of Nyquist. The slice is empty for a direct design.
    pub fn extrema(&self) -> &[S] {
        &self.extrema
    }
    /// Returns the estimated one-norm condition number of the normal equations
    /// of a least-squares design, or `None` for the other designs.
    pub fn condition_estimate(&self) -> Option<S> {
        self.condition
    }
    /// Computes the frequency response `H(f) = Σᵢ h[i] e^(−iπfi)` at
    /// `f = frequency`, in multiples of Nyquist.
    ///
    /// The sum is evaluated directly in `O(n)` operations. Its real and
    /// imaginary parts are accumulated exactly and rounded once.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `frequency` lies outside `[0, 1]`.
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` if the response overflows.
    pub fn response(&self, frequency: S) -> Result<Complex<S>, SciOxideError> {
        self.response_at(frequency, scioxide_core::argument!("frequency"))
    }
    /// Frequency response; `argument` names the parameter that supplied
    /// `frequency`.
    fn response_at(&self, frequency: S, argument: Argument) -> Result<Complex<S>, SciOxideError> {
        frequency_check(frequency, argument)?;
        let mut re = scioxide_core::RangeAccumulator::new();
        let mut im = scioxide_core::RangeAccumulator::new();
        for (i, &h) in self.taps.iter().enumerate() {
            let phase = S::PI * frequency * scalar::<S>(i)?;
            let (sin, cos) = phase.sin_cos();
            re.add_product(h, cos)?;
            im.subtract_product(h, sin)?;
        }
        Ok(Complex::new(re.finish()?, im.finish()?))
    }
    /// Returns a [`PolyphasePlan`](crate::PolyphasePlan) that filters real or
    /// complex samples with these taps.
    ///
    /// The plan holds the taps converted to the sample type `T`, and it filters
    /// records, lanes along an axis and streams without rate change; on a
    /// record it returns the full convolution, as
    /// [`PolyphasePlan::fir`](crate::PolyphasePlan::fir) describes. The
    /// conversion costs `O(n)` for `n` taps.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`PolyphasePlan::fir`](crate::PolyphasePlan::fir).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{FirBand, FirNormalization, FirSymmetry, WindowKind, design_fir_window};
    /// let band = FirBand::new(0.0_f64, 0.5, 1.0, 1.0);
    /// let (symmetry, normalization) = (FirSymmetry::Even, FirNormalization::None);
    /// let design = design_fir_window(9, symmetry, &[band], WindowKind::Hann, normalization)?;
    /// // Filtering a unit impulse returns the taps.
    /// let impulse = design.to_polyphase::<f64>()?.resample(&[1.0])?;
    /// let taps = design.coefficients();
    /// assert!(impulse.iter().zip(taps).all(|(y, h)| (y - h).abs() < 1e-15));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_polyphase<T: scioxide_core::NumericScalar<Real = S>>(
        &self,
    ) -> Result<crate::PolyphasePlan<T>, SciOxideError> {
        let mut taps = try_filled(
            self.taps.len(),
            T::ZERO,
            scioxide_core::quantity!("fir_execution_taps"),
        )?;
        taps.iter_mut()
            .zip(&self.taps)
            .for_each(|(t, &h)| *t = T::from_real(h));
        crate::PolyphasePlan::fir(&taps)
    }
}
/// Iteration limit, grid density and tolerance of the Remez exchange.
///
/// The defaults are 64 exchange iterations, a grid density of 32 and a
/// tolerance of `16ε` absolute and `√ε` relative on the spread of the
/// alternating weighted errors, where `ε` is the machine epsilon of `S`. The
/// absolute part is at the rounding level of a unit-scale response, so a small
/// ripple that is still above the minimax optimum does not count as
/// convergence. The grid density is the number of grid points per independent
/// coefficient per unit of normalized frequency.
///
/// [`design_fir_remez`] returns `InvalidInput` for zero iterations or a grid
/// density below 16.
///
/// ```
/// use scioxide_signal::RemezOptions;
/// let options = RemezOptions::<f64>::new().with_max_iterations(8).with_grid_density(16);
/// assert_eq!((options.max_iterations(), options.grid_density()), (8, 16));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RemezOptions<S: RealScalar> {
    max_iterations: usize,
    grid_density: usize,
    tolerance: ResidualTolerance<S>,
}
impl<S: RealScalar> Default for RemezOptions<S> {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            grid_density: 32,
            tolerance: default_tolerance(),
        }
    }
}
/// `16*epsilon` absolute, `sqrt(epsilon)` relative.
fn default_tolerance<S: RealScalar>() -> ResidualTolerance<S> {
    let sixteen = S::TWO * S::TWO * S::TWO * S::TWO;
    let tolerance = scioxide_core::Tolerance::from_sqrt_epsilon();
    ResidualTolerance::from_tolerance(
        tolerance
            .with_absolute(sixteen * S::EPSILON)
            .unwrap_or(tolerance),
    )
}
impl<S: RealScalar> RemezOptions<S> {
    /// Creates the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the maximum number of exchange iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    /// Sets the grid density.
    pub const fn with_grid_density(mut self, grid_density: usize) -> Self {
        self.grid_density = grid_density;
        self
    }
    /// Sets the absolute and relative tolerance on the spread of the
    /// alternating weighted errors.
    pub const fn with_tolerance(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns the maximum number of exchange iterations.
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }
    /// Returns the grid points per independent coefficient per unit of
    /// normalized frequency.
    pub const fn grid_density(self) -> usize {
        self.grid_density
    }
    /// Returns the tolerance on the spread of the alternating weighted errors.
    pub const fn tolerance(self) -> ResidualTolerance<S> {
        self.tolerance
    }
}
/// Reusable rank-aware Householder solve of a small dense design system.
struct DesignSolve<S: RealScalar> {
    factor: HouseholderDecomposition<S>,
    workspace: LinalgWorkspace<'static, S>,
    solution: Vector<S>,
}
impl<S: RealScalar> DesignSolve<S> {
    fn new(rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(
            &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::factor(
                    rows,
                    cols,
                    HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?,
        )?;
        workspace.try_reserve(
            &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                    rows,
                    rows,
                    scioxide_linalg::HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        Ok(Self {
            factor: HouseholderDecomposition::empty(),
            workspace,
            solution: Vector::zeros(cols)?,
        })
    }
    /// Solve into `self.solution` and return the one-norm condition estimate.
    fn solve(
        &mut self,
        matrix: &Matrix<S>,
        rhs: &Vector<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<S, SciOxideError> {
        HouseholderDecomposition::factor_into(
            matrix,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut self.factor,
            &mut self.workspace,
            execution,
        )?;
        if !self
            .factor
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
            .is_full_rank()
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "rank-deficient design",
            ));
        }
        let report = self.factor.solve_into(
            rhs,
            &(scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()),
            self.solution.view_mut(),
            &mut self.workspace,
            execution,
        )?;
        if !report.termination().is_success() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "design solve rejected its solution",
            ));
        }
        finite(
            S::ONE
                / self
                    .factor
                    .condition_estimate_scoped(
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        &mut self.workspace,
                        execution,
                    )?
                    .reciprocal()
                    .value(),
        )
    }
}
/// Requires a normalized frequency in [0, 1] supplied through `argument`.
fn frequency_check<S: RealScalar>(f: S, argument: Argument) -> Result<(), SciOxideError> {
    check_real_slice_finite(&[f])?;
    if f < S::ZERO || f > S::ONE {
        return Err(invalid(argument, "normalized frequency must lie in [0, 1]"));
    }
    Ok(())
}
fn dimensions(n: usize, symmetry: FirSymmetry) -> Result<usize, SciOxideError> {
    if n == 0 {
        return Err(empty(
            scioxide_core::argument!("n"),
            "tap count must be positive",
        ));
    }
    if n == 1 && symmetry == FirSymmetry::Odd {
        return Err(inconsistent(
            scioxide_core::argument!("n"),
            "odd symmetry needs at least two taps",
        ));
    }
    checked_mul(
        n,
        core::mem::size_of::<usize>(),
        scioxide_core::quantity!("fir_dimensions"),
    )?;
    Ok(if symmetry == FirSymmetry::Even {
        n / 2 + n % 2
    } else {
        n / 2
    })
}
fn forced_zero<S: RealScalar>(n: usize, symmetry: FirSymmetry, f: S) -> bool {
    (f == S::ZERO && symmetry == FirSymmetry::Odd)
        || (f == S::ONE
            && ((symmetry == FirSymmetry::Even && n.is_multiple_of(2))
                || (symmetry == FirSymmetry::Odd && !n.is_multiple_of(2))))
}
fn validate<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
) -> Result<usize, SciOxideError> {
    let m = dimensions(n, symmetry)?;
    let bands_argument = scioxide_core::argument!("bands");
    if bands.is_empty() {
        return Err(empty(bands_argument, "at least one band is required"));
    }
    let mut previous = None;
    for b in bands {
        check_real_slice_finite(&[b.lower, b.upper, b.desired_lower, b.desired_upper, b.weight])?;
        frequency_check(b.lower, bands_argument)?;
        frequency_check(b.upper, bands_argument)?;
        if b.lower >= b.upper || b.weight <= S::ZERO || previous.is_some_and(|p| b.lower < p) {
            return Err(invalid(
                bands_argument,
                "bands need lower < upper, positive weights and ascending order",
            ));
        }
        for (f, d) in [(b.lower, b.desired_lower), (b.upper, b.desired_upper)] {
            if forced_zero(n, symmetry, f) && d != S::ZERO {
                return Err(inconsistent(
                    bands_argument,
                    "desired response must vanish where the symmetry forces a zero",
                ));
            }
        }
        previous = Some(b.upper);
    }
    Ok(m)
}
fn from_amplitudes<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    a: &[S],
) -> Result<FirDesign<S>, SciOxideError> {
    let mut taps = try_filled(n, S::ZERO, scioxide_core::quantity!("fir_taps"))?;
    fill_taps(&mut taps, symmetry, a)?;
    Ok(FirDesign {
        taps,
        symmetry,
        termination: FirTermination::Completed,
        iterations: 0,
        ripple: None,
        extrema: Vec::new(),
        condition: None,
    })
}
fn fill_taps<S: RealScalar>(
    taps: &mut [S],
    symmetry: FirSymmetry,
    a: &[S],
) -> Result<(), SciOxideError> {
    let n = taps.len();
    let two = S::TWO;
    for (j, &v) in a.iter().enumerate() {
        let center = symmetry == FirSymmetry::Even && !n.is_multiple_of(2) && j == 0;
        let left = if center {
            n / 2
        } else if !n.is_multiple_of(2) && symmetry == FirSymmetry::Even {
            n / 2 - j
        } else {
            n / 2 - 1 - j
        };
        let h = finite(if center { v } else { v / two })?;
        taps[left] = h;
        taps[n - 1 - left] = if symmetry == FirSymmetry::Even { h } else { -h };
    }
    Ok(())
}
fn normalize<S: RealScalar>(
    design: &mut FirDesign<S>,
    normalization: FirNormalization<S>,
) -> Result<(), SciOxideError> {
    let normalization_argument = scioxide_core::argument!("normalization");
    if let FirNormalization::At { frequency, gain } = normalization {
        frequency_check(frequency, normalization_argument)?;
        check_real_slice_finite(&[gain])?;
        let amplitude = if frequency == S::ZERO && design.symmetry == FirSymmetry::Even {
            match scioxide_core::compensated_reduce(design.taps.iter().map(|&h| [h]))
                .and_then(|[sum]| sum.total())
            {
                Ok(sum) => sum,
                Err(_) => crate::util::sum_products(design.taps.iter().map(|&h| (h, S::ONE)))?,
            }
        } else {
            let h = design.response_at(frequency, normalization_argument)?;
            let phase = S::PI * frequency * scalar::<S>(design.taps.len() - 1)? / (S::TWO);
            let (sin, cos) = phase.sin_cos();
            if design.symmetry == FirSymmetry::Even {
                h.re * cos - h.im * sin
            } else {
                h.re * sin + h.im * cos
            }
        };
        if amplitude == S::ZERO || forced_zero(design.taps.len(), design.symmetry, frequency) {
            return Err(SciOxideError::domain_error(
                normalization_argument,
                scioxide_core::DomainReason::Undefined,
                "response vanishes at the normalization frequency",
            ));
        }
        let scale = finite(gain / amplitude)?;
        for h in &mut design.taps {
            *h = finite(*h * scale)?;
        }
    }
    Ok(())
}
// Integral of cos(pi*k*f), evaluated in centered coordinates to avoid subtracting close sines.
fn cosine_integral<S: RealScalar>(k: S, lo: S, hi: S) -> Result<S, SciOxideError> {
    let two = S::TWO;
    let width = hi - lo;
    if lo == S::ZERO {
        return Ok(width * scioxide_core::sincpi(k * hi)?);
    }
    Ok(width * (S::PI * k * (hi + lo) / two).cos() * scioxide_core::sincpi(k * width / two)?)
}
fn sine_integral<S: RealScalar>(k: S, lo: S, hi: S) -> Result<S, SciOxideError> {
    let two = S::TWO;
    let width = hi - lo;
    Ok(width * (S::PI * k * (hi + lo) / two).sin() * scioxide_core::sincpi(k * width / two)?)
}
