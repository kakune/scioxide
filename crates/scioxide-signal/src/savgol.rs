//! Savitzky–Golay smoothing and differentiation filters.
use crate::util::{
    capacity, empty, finite, inconsistent, invalid, scalar_from_usize_rounded as scalar,
    unsupported,
};
use crate::util::{inputs, lane_requirements, size};
use crate::{SignalLayout, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy,
};
/// Treatment of outputs whose window extends past the ends of the record.
#[derive(Clone, Copy, Debug)]
pub enum SavitzkyGolayBoundary {
    /// Fits the polynomial to the first (or last) complete window and evaluates
    /// it at the output position. The record must be at least one window long.
    Polynomial,
    /// Extends the record by the given [`SignalBoundary`](crate::SignalBoundary)
    /// rule and applies the interior filter.
    Extend(crate::SignalBoundary),
}
/// Derivative order, sample spacing and evaluation point of a Savitzky–Golay
/// filter. The window length and the polynomial degree are passed to the plan
/// separately.
///
/// The defaults are derivative order zero (smoothing), unit spacing,
/// evaluation at the window midpoint, which needs an odd window, and
/// sequential execution. The setters do not validate.
/// [`SavitzkyGolayPlan::new`] checks the options against the window length and
/// the degree.
///
/// ```
/// use scioxide_signal::SavitzkyGolayOptions;
/// let options = SavitzkyGolayOptions::new()
///     .with_derivative(1)
///     .with_spacing(0.5_f64);
/// assert_eq!((options.derivative(), options.spacing()), (1, 0.5));
/// assert_eq!(SavitzkyGolayOptions::<f64>::default().position(), None);
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SavitzkyGolayOptions<S: RealScalar> {
    derivative: usize,
    spacing: S,
    position: Option<S>,
    policy: ExecutionPolicy,
}
impl<S: RealScalar> Default for SavitzkyGolayOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> SavitzkyGolayOptions<S> {
    /// Returns the default options listed at [`SavitzkyGolayOptions`].
    pub const fn new() -> Self {
        Self {
            derivative: 0,
            spacing: S::ONE,
            position: None,
            policy: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the derivative order `d`, at most the polynomial degree.
    pub const fn with_derivative(mut self, derivative: usize) -> Self {
        self.derivative = derivative;
        self
    }
    /// Sets the sample spacing `h > 0`, which scales the `d`-th derivative by `h⁻ᵈ`.
    pub const fn with_spacing(mut self, spacing: S) -> Self {
        self.spacing = spacing;
        self
    }
    /// Sets the evaluation point `t ∈ [0, w − 1]` for window length `w`, in
    /// samples from the start of the window, instead of the midpoint.
    pub const fn with_position(mut self, position: S) -> Self {
        self.position = Some(position);
        self
    }
    /// Sets the execution policy for the lanes of
    /// [`SavitzkyGolayPlan::filter_axis_into`].
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the derivative order `d`.
    pub const fn derivative(self) -> usize {
        self.derivative
    }
    /// Returns the sample spacing `h`.
    pub const fn spacing(self) -> S {
        self.spacing
    }
    /// Returns the evaluation point, or `None` for the window midpoint.
    pub const fn position(self) -> Option<S> {
        self.position
    }
    /// Returns the execution policy for axis lanes.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
}
/// Smooths a record with a Savitzky–Golay filter.
///
/// Output `i` is the value at sample `i` of the least-squares polynomial of
/// degree `polynomial_degree` fitted to the `window_length` samples centered
/// on it. This is [`SavitzkyGolayPlan::new`] with
/// [`SavitzkyGolayOptions::default`] (no derivative, unit spacing, evaluation
/// at the window midpoint, so the window length is odd) followed by
/// [`SavitzkyGolayPlan::filter`]. The boundary is fixed to
/// [`SavitzkyGolayBoundary::Polynomial`]: the boundary has no neutral default,
/// and this one assumes nothing outside the record, because it evaluates the
/// polynomial fitted to the first or last complete window at the edge samples
/// (Gorry, 1990). A nonempty record must therefore be at least one window long,
/// and an empty record gives an empty result. For window length `w` and degree
/// `p`, the design costs `O(w p²)` and filtering `O(w)` per sample, and each
/// output meets the product-sum bound of [Accuracy](crate#accuracy). Real
/// coefficients apply to real or complex samples. The plan takes derivatives,
/// other boundaries and repeated records. This function runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `window_length` is zero.
/// - `InvalidInput` (`OutOfRange`) if `polynomial_degree` is not below
///   `window_length`.
/// - `InvalidInput` (`UnsupportedConfiguration`) if `window_length` is even.
/// - `InvalidInput` (`Inconsistent`) if `input` is nonempty and shorter than
///   the window.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` if the design matrix is numerically rank
///   deficient, a coefficient overflows or underflows to zero, or an output
///   overflows.
/// - `AllocationFailure` if the filter, the workspace or the result cannot be
///   allocated.
///
/// # Examples
///
/// ```
/// // A quadratic fit over five samples reproduces a parabola, edges included.
/// let input: Vec<f64> = (0..9).map(|k| (k as f64 - 3.0).powi(2)).collect();
/// let smoothed = scioxide_signal::savitzky_golay(&input, 5, 2)?;
/// for (y, x) in smoothed.iter().zip(&input) {
///     assert!((y - x).abs() < 1e-10);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by
///   simplified least squares procedures", *Anal. Chem.* 36(8), 1627–1639,
///   1964.
/// - P. A. Gorry, "General least-squares smoothing and differentiation by the
///   convolution (Savitzky–Golay) method", *Anal. Chem.* 62(6), 570–573, 1990.
pub fn savitzky_golay<T: NumericScalar>(
    input: &[T],
    window_length: usize,
    polynomial_degree: usize,
) -> Result<Vec<T>, SciOxideError> {
    SavitzkyGolayPlan::<T::Real>::new(
        window_length,
        polynomial_degree,
        SavitzkyGolayOptions::default(),
    )?
    .filter(input, SavitzkyGolayBoundary::Polynomial)
}
/// Savitzky–Golay filter: least-squares polynomial fits over a sliding window.
///
/// For each output sample, the filter fits a polynomial of degree `p` to the
/// `w` samples of the window by least squares and returns the `d`-th derivative
/// of the fit at the evaluation point `t`. This is a correlation with fixed
/// coefficients `c`:
///
/// ```text
/// y[i] = Σⱼ c[j] x[i − ⌊t⌋ + j],   j = 0, …, w − 1,
/// ```
///
/// so `y[i]` estimates the `d`-th derivative at sample position `i + t − ⌊t⌋`.
/// Real coefficients apply to real or complex samples.
///
/// The fit uses a column-pivoted Householder QR factorization of the `w × (p+1)`
/// monomial design matrix on the window abscissae scaled to `[−1, 1]`.
/// Preparation costs `O(w p²)` and stores `O(w p)` values. Filtering costs
/// `O(w)` per sample. Its accuracy follows the product-sum bound in
/// [Accuracy](crate#accuracy).
///
/// # References
///
/// A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by
/// simplified least squares procedures", *Anal. Chem.* 36(8), 1627–1639, 1964.
#[derive(Clone, Debug)]
pub struct SavitzkyGolayPlan<S: RealScalar> {
    window_length: usize,
    polynomial_degree: usize,
    options: SavitzkyGolayOptions<S>,
    anchor: usize,
    coefficients: Vec<S>,
    // Row `k` maps a window to coefficient `k` of its fitted scaled monomial polynomial.
    fit: Vec<S>,
    // Row `i` weights the fitted coefficients into the output at window position `i`.
    evaluation: Vec<S>,
    condition: S,
}
impl<S: RealScalar> SavitzkyGolayPlan<S> {
    /// Prepares the filter that fits polynomials of degree `polynomial_degree`
    /// to windows of `window_length` samples.
    ///
    /// Runs [`SavitzkyGolayPlan::build`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// ```
    /// use scioxide_signal::{SavitzkyGolayBoundary, SavitzkyGolayOptions, SavitzkyGolayPlan};
    /// let plan = SavitzkyGolayPlan::new(3, 1, SavitzkyGolayOptions::new())?;
    /// let smoothed = plan.filter(&[1.0_f64, 2.0, 3.0], SavitzkyGolayBoundary::Polynomial)?;
    /// // A linear fit reproduces a linear record exactly.
    /// assert!(
    ///     smoothed
    ///         .iter()
    ///         .zip([1.0, 2.0, 3.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-12)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`SavitzkyGolayPlan::build`].
    pub fn new(
        window_length: usize,
        polynomial_degree: usize,
        options: SavitzkyGolayOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            window_length,
            polynomial_degree,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares the filter that fits polynomials of degree `polynomial_degree`
    /// to windows of `window_length` samples, under an execution context.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `window_length` is zero.
    /// - `InvalidInput` (`OutOfRange`) if `polynomial_degree` is not less than
    ///   `window_length`, the derivative order exceeds the degree, the spacing
    ///   is not positive, or the evaluation point lies outside
    ///   `[0, window_length − 1]`.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the midpoint is
    ///   requested for an even `window_length`.
    /// - `NonFinite` if the spacing or the evaluation point is NaN or infinite.
    /// - `NumericalInstability` if the design matrix is numerically rank
    ///   deficient, or a coefficient overflows or underflows to zero.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn build(
        window_length: usize,
        polynomial_degree: usize,
        options: SavitzkyGolayOptions<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        execution.check()?;
        let n = window_length;
        let p = polynomial_degree;
        crate::util::check_real_slice_finite(&[options.spacing])?;
        if n == 0 {
            return Err(empty(
                scioxide_core::argument!("window_length"),
                "window length must be positive",
            ));
        }
        if p >= n {
            return Err(invalid(
                scioxide_core::argument!("polynomial_degree"),
                "polynomial degree must be below the window length",
            ));
        }
        if options.derivative > p {
            return Err(invalid(
                scioxide_core::argument!("options.derivative"),
                "derivative order must not exceed the polynomial degree",
            ));
        }
        if options.spacing <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.spacing"),
                "sample spacing must be positive",
            ));
        }
        if options.position.is_none() && n.is_multiple_of(2) {
            return Err(unsupported(
                scioxide_core::argument!("window_length"),
                "a midpoint position requires an odd window length",
            ));
        }
        let two = S::TWO;
        let center = scalar::<S>(n - 1)? / two;
        let scale = center.max(S::ONE);
        let position = options.position.unwrap_or(center);
        crate::util::check_real_slice_finite(&[position])?;
        if position < S::ZERO || position > scalar::<S>(n - 1)? {
            return Err(invalid(
                scioxide_core::argument!("options.position"),
                "position must lie within the window",
            ));
        }
        let anchor = (0..n)
            .take_while(|&i| scalar::<S>(i).is_ok_and(|s| s <= position))
            .last()
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("options.position"),
                    "position has no anchoring sample",
                )
            })?;
        let fraction = position - scalar::<S>(anchor)?;
        let mut a = Matrix::zeros(n, p + 1)?;
        for i in 0..n {
            let x = (scalar::<S>(i)? - center) / scale;
            let mut power = S::ONE;
            for j in 0..=p {
                a.view_mut().set(i, j, power)?;
                power *= x;
            }
        }
        let mut factor = HouseholderDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        HouseholderDecomposition::factor_into(
            &a,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut workspace,
            execution,
        )?;
        if !factor
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
            .is_full_rank()
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "polynomial fit has lost numerical rank",
            ));
        }
        let condition = finite(
            S::ONE
                / factor
                    .condition_estimate_scoped(
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        &mut workspace,
                        execution,
                    )?
                    .reciprocal()
                    .value(),
        )?;
        let terms = p + 1;
        // Thin orthonormal factor of `A[:, permutation] = Q R`.
        let sequential = scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential);
        let mut seed = Matrix::zeros(n, terms)?;
        for k in 0..terms {
            seed.view_mut().set(k, k, S::ONE)?;
        }
        let mut q = Matrix::zeros(n, terms)?;
        workspace.try_reserve(&HouseholderDecomposition::<S>::workspace_requirements(
            &scioxide_linalg::HouseholderWorkspaceRequest::apply_q(
                n,
                terms,
                HouseholderOrientation::Qr,
                terms,
                &sequential,
            ),
        )?)?;
        factor.apply_q_into(
            seed.view(),
            scioxide_linalg::StructuredOperation::Normal,
            &sequential,
            q.view_mut(),
            &mut workspace,
            execution,
        )?;
        let mut triangle = try_filled(
            size(terms, terms)?,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_triangle"),
        )?;
        let packed = factor.packed_factors();
        for (j, row) in triangle.chunks_exact_mut(terms).enumerate() {
            for (l, value) in row.iter_mut().enumerate().skip(j) {
                *value = packed.get(j, l)?;
            }
        }
        let permutation = factor.permutation();
        let mut coefficients = try_filled(
            n,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_coefficients"),
        )?;
        let mut fit = try_filled(
            size(terms, n)?,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_fit"),
        )?;
        let mut pivoted = try_filled(
            terms,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_fit"),
        )?;
        let mut polynomial = try_filled(
            terms,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_fit"),
        )?;
        // The polynomial fit to unit impulse `i` is column `i` of the
        // pseudoinverse `P R⁻¹ Qᵀ`: back substitution against row `i` of Q.
        for (i, row) in q.as_slice().chunks_exact(terms).enumerate() {
            if i.is_multiple_of(64) {
                execution.check()?;
            }
            for j in (0..terms).rev() {
                let r = &triangle[j * terms..(j + 1) * terms];
                let sum = (j + 1..terms).fold(row[j], |sum, l| sum - r[l] * pivoted[l]);
                pivoted[j] = finite(sum / r[j])?;
            }
            for (&target, &value) in permutation.iter().zip(&pivoted) {
                polynomial[target] = value;
            }
            coefficients[i] = evaluate_derivative(
                &polynomial,
                options.derivative,
                (position - center) / scale,
                scale,
                options.spacing,
            )?;
            for (k, &value) in polynomial.iter().enumerate() {
                fit[k * n + i] = value;
            }
        }
        // Output weight of each scaled monomial at every window position.
        let mut evaluation = try_filled(
            size(n, terms)?,
            S::ZERO,
            scioxide_core::quantity!("savitzky_golay_evaluation"),
        )?;
        for (i, row) in evaluation.chunks_exact_mut(terms).enumerate() {
            if i.is_multiple_of(64) {
                execution.check()?;
            }
            let x = (scalar::<S>(i)? + fraction - center) / scale;
            for k in options.derivative..terms {
                polynomial.fill(S::ZERO);
                polynomial[k] = S::ONE;
                row[k] = evaluate_derivative(
                    &polynomial,
                    options.derivative,
                    x,
                    scale,
                    options.spacing,
                )?;
            }
        }
        Ok(Self {
            window_length,
            polynomial_degree,
            options,
            anchor,
            coefficients,
            fit,
            evaluation,
            condition,
        })
    }
    /// Returns the interior coefficients `c`, applied to the window samples in
    /// increasing time order.
    pub fn coefficients(&self) -> &[S] {
        &self.coefficients
    }
    /// Returns the window length `w`.
    pub fn window_length(&self) -> usize {
        self.window_length
    }
    /// Returns the polynomial degree `p`.
    pub fn polynomial_degree(&self) -> usize {
        self.polynomial_degree
    }
    /// Returns the options.
    pub fn options(&self) -> SavitzkyGolayOptions<S> {
        self.options
    }
    /// Returns an estimate of the one-norm condition number of the scaled design
    /// matrix.
    ///
    /// The estimate indicates how sensitive the fit is to rounding. It is not a
    /// bound on the error of the filter output.
    pub fn condition_estimate(&self) -> S {
        self.condition
    }
    /// Returns the workspace size for one record or for every lane along an axis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the axis of an axis layout is out of range.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        layout: SignalLayout<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        match layout {
            SignalLayout::Contiguous(n) => Ok(SignalRequirements::values(checked_add(
                n,
                self.polynomial_degree + 1,
                scioxide_core::quantity!("savitzky_golay_workspace"),
            )?)),
            layout => lane_requirements(layout, 3),
        }
    }
    /// Allocates a workspace for `layout`.
    ///
    /// # Errors
    ///
    /// The errors of [`SavitzkyGolayPlan::workspace_requirements`], and
    /// `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
        layout: SignalLayout<'_>,
    ) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(layout)?)
    }
    /// Filters a record and returns the result.
    ///
    /// # Errors
    ///
    /// The errors of [`SavitzkyGolayPlan::filter_into`], and `AllocationFailure`
    /// if the result or workspace cannot be allocated.
    pub fn filter<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        boundary: SavitzkyGolayBoundary,
    ) -> Result<Vec<T>, SciOxideError> {
        let mut output = try_filled(
            input.len(),
            T::ZERO,
            scioxide_core::quantity!("savitzky_golay_output"),
        )?;
        let mut workspace = self.workspace(SignalLayout::Contiguous(input.len()))?;
        self.filter_into(
            input,
            boundary,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Filters a record into the first `input.len()` entries of `output`.
    ///
    /// Empty records are valid.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the boundary is
    ///   [`Polynomial`](SavitzkyGolayBoundary::Polynomial) and the record is
    ///   nonempty but shorter than the window.
    /// - `DimensionMismatch` if `output` is shorter than `input` or the workspace
    ///   is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if an output overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn filter_into<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        boundary: SavitzkyGolayBoundary,
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        capacity(input.len(), output.len())?;
        let requirements = self.workspace_requirements(SignalLayout::Contiguous(input.len()))?;
        let (staged, fitted) = workspace
            .buffers(&requirements)?
            .values
            .split_at_mut(input.len());
        inputs(input)?;
        self.check_record(input.len(), boundary)?;
        execution.check()?;
        self.run(input, staged, fitted, boundary, execution)?;
        execution.check()?;
        output[..input.len()].copy_from_slice(staged);
        Ok(())
    }
    /// Reject polynomial-boundary records shorter than the window.
    fn check_record(
        &self,
        len: usize,
        boundary: SavitzkyGolayBoundary,
    ) -> Result<(), SciOxideError> {
        if len != 0
            && matches!(boundary, SavitzkyGolayBoundary::Polynomial)
            && len < self.window_length
        {
            return Err(inconsistent(
                scioxide_core::argument!("input"),
                "polynomial boundary needs a record at least as long as the window",
            ));
        }
        Ok(())
    }
    /// Evaluate every output of a validated record into `output`.
    ///
    /// A polynomial boundary fits the first/last complete window once into
    /// `fitted` (`p+1` window product sums) and evaluates that polynomial at
    /// each edge output (`p+1` products), so a record end costs `O(w*p)`.
    fn run<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        output: &mut [T],
        fitted: &mut [T],
        boundary: SavitzkyGolayBoundary,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.window_length;
        let terms = self.polynomial_degree + 1;
        let len = input.len();
        let mut window = None;
        for (i, y) in output[..len].iter_mut().enumerate() {
            execution.check()?;
            let start = i as i128 - self.anchor as i128;
            if matches!(boundary, SavitzkyGolayBoundary::Polynomial)
                && (start < 0 || start + n as i128 > len as i128)
            {
                let offset = start.clamp(0, (len - n) as i128) as usize;
                if window != Some(offset) {
                    for (a, row) in fitted[..terms].iter_mut().zip(self.fit.chunks_exact(n)) {
                        *a = crate::util::contiguous_products::<T, S, false>(
                            &input[offset..offset + n],
                            row,
                            T::from_real,
                        )?;
                    }
                    window = Some(offset);
                }
                let position = i - offset;
                *y = crate::util::contiguous_products::<T, S, false>(
                    &fitted[..terms],
                    &self.evaluation[position * terms..(position + 1) * terms],
                    T::from_real,
                )?;
                continue;
            }
            if start >= 0 && start + n as i128 <= len as i128 {
                *y = crate::util::contiguous_products::<T, S, false>(
                    &input[start as usize..start as usize + n],
                    &self.coefficients,
                    T::from_real,
                )?;
                continue;
            }
            *y = crate::util::working_sum_products(self.coefficients.iter().enumerate().map(
                |(j, &c)| {
                    let index = start + j as i128;
                    let v = match boundary {
                        SavitzkyGolayBoundary::Polynomial => input[index as usize],
                        SavitzkyGolayBoundary::Extend(b) => {
                            crate::util::boundary_index(index, len, b).map_or(T::ZERO, |k| input[k])
                        }
                    };
                    (T::from_real(c), v)
                },
            ))?;
        }
        Ok(())
    }
    /// Filters every lane along `axis` of an N-D array.
    ///
    /// Lanes run in parallel under the execution policy of the options.
    ///
    /// # Errors
    ///
    /// The errors of [`SavitzkyGolayPlan::filter_into`] for each lane, and
    /// `InvalidInput` if `axis` is out of range or `DimensionMismatch` if the
    /// input and output shapes differ.
    pub fn filter_axis_into<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        boundary: SavitzkyGolayBoundary,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::util::axis_filter(
            input,
            output,
            axis,
            workspace,
            3,
            self.options.policy,
            execution,
            |x, y, fitted| {
                inputs(x)?;
                self.check_record(x.len(), boundary)?;
                self.run(x, y, fitted, boundary, execution)
            },
        )
    }
}
fn evaluate_derivative<S: RealScalar>(
    polynomial: &[S],
    derivative: usize,
    x: S,
    scale: S,
    spacing: S,
) -> Result<S, SciOxideError> {
    use scioxide_core::Scaled;
    use scioxide_poly::backend::polynomial_basis::newton_value_and_derivative_from_iter;
    let coefficient = |k: usize| -> Result<Scaled<S>, SciOxideError> {
        (0..derivative).try_fold(Scaled::from_value(polynomial[k])?, |c, j| {
            c.mul(Scaled::from_value(scalar::<S>(k - j)?)?)
        })
    };
    let last = polynomial.len() - 1;
    let (value, _) = newton_value_and_derivative_from_iter(
        (derivative..last).map(|k| Ok((coefficient(k)?, Scaled::ZERO))),
        coefficient(last)?,
        Scaled::from_value(x)?,
        Scaled::ZERO,
        |v, x, node| v.mul(x.sub(node)?),
        |a, b| a.add(b),
    )?;
    let mut value = value.round()?;
    for _ in 0..derivative {
        let next = finite(value / scale / spacing)?;
        if value != S::ZERO && next == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "derivative coefficient underflow",
            ));
        }
        value = next;
    }
    Ok(value)
}
