//! Symmetric recursive filters and B-spline interpolation of uniform samples.
use crate::util::{
    axis_filter, capacity, check, empty, finite, inputs, invalid, lane_requirements,
    scalar_from_usize_rounded as scalar, size, zeros,
};
use crate::{SignalExecutionOptions, SignalLayout, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::{
    ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideError,
};

/// Zero-phase recursive filter with symmetric boundary extension.
///
/// The transfer function is
///
/// ```text
/// H(z) = g / (A(z) A(z⁻¹)),
/// A(z) = 1 − p z⁻¹                         (first order),
/// A(z) = 1 − 2r cos θ z⁻¹ + r² z⁻²         (second order),
/// ```
///
/// with gain `g`, real pole `p` or complex-conjugate poles `r e^(±iθ)`. The
/// filter runs a causal pass of `1/A(z)` and an anticausal pass of
/// `1/A(z⁻¹)`. The record `x[0..n]` is extended by half-sample symmetric
/// reflection (`x[−1] = x[0]`, `x[n] = x[n−1]`), and each pass starts from the
/// exact steady state of that periodic extension, so there are no start-up
/// transients. Filtering costs `O(n)` operations. The filter holds the
/// validated coefficients and computes nothing at construction.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{SignalExecutionOptions, SignalLayout, SymmetricFilter};
/// let filter = SymmetricFilter::first_order(0.5_f64, 0.25, SignalExecutionOptions::new())?;
/// let mut workspace = filter.workspace(SignalLayout::Contiguous(4))?;
/// let mut output = [0.0_f64; 4];
/// let cx = ExecutionContext::sequential_native();
/// filter.filter_into(&[1.0; 4], &mut output, &mut workspace, cx)?;
/// // H(1) = 0.25 / (1 − 0.5)² = 1, so a constant record is unchanged.
/// assert!(output.iter().all(|y| (y - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SymmetricFilter<S: RealScalar> {
    a: S,
    b: S,
    gain: S,
    options: SignalExecutionOptions,
}
impl<S: RealScalar> SymmetricFilter<S> {
    /// Prepares a first-order filter with real pole `p` and gain `g`.
    ///
    /// The execution policy of `options` applies to
    /// [`SymmetricFilter::filter_axis_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `|pole| ≥ 1`.
    /// - `NonFinite` if `pole` or `gain` is NaN or infinite.
    pub fn first_order(
        pole: S,
        gain: S,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        crate::util::check_real_slice_finite(&[pole, gain])?;
        if pole.abs() >= S::ONE {
            return Err(invalid(
                scioxide_core::argument!("pole"),
                "pole magnitude must be below one",
            ));
        }
        Ok(Self {
            a: pole,
            b: S::ZERO,
            gain,
            options,
        })
    }
    /// Prepares a second-order filter with poles `r e^(±iθ)` and gain `g`.
    ///
    /// `gain` is the numerator `g` of the two-sided transfer function `H(z)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `radius` is outside `[0, 1)` or `angle` is outside `[0, π]`.
    /// - `NonFinite` if `radius`, `angle` or `gain` is NaN or infinite.
    pub fn second_order(
        radius: S,
        angle: S,
        gain: S,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        crate::util::check_real_slice_finite(&[radius, angle, gain])?;
        if radius < S::ZERO || radius >= S::ONE {
            return Err(invalid(
                scioxide_core::argument!("radius"),
                "pole radius must lie in [0, 1)",
            ));
        }
        if angle < S::ZERO || angle > S::PI {
            return Err(invalid(
                scioxide_core::argument!("angle"),
                "pole angle must lie in [0, pi]",
            ));
        }
        Ok(Self {
            a: (S::TWO) * radius * angle.cos(),
            b: -radius * radius,
            gain,
            options,
        })
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
            SignalLayout::Contiguous(n) => Ok(SignalRequirements::values(size(n, 4)?)),
            layout => lane_requirements(layout, 6),
        }
    }
    /// Allocates a workspace for `layout`.
    ///
    /// # Errors
    ///
    /// The errors of [`SymmetricFilter::workspace_requirements`], and
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
    /// The errors of [`SymmetricFilter::filter_into`], and
    /// `AllocationFailure` if the result or workspace cannot be allocated.
    pub fn filter<T: NumericScalar<Real = S>>(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        let mut out = zeros(input.len())?;
        let mut workspace = self.workspace(SignalLayout::Contiguous(input.len()))?;
        self.filter_into(
            input,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
    /// Filters a record into the first `input.len()` entries of `output`.
    ///
    /// Empty records are valid.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is shorter than `input` or the workspace
    ///   is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a value overflows, or the steady-state system
    ///   of the boundary extension is singular in floating point.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn filter_into<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        capacity(input.len(), output.len())?;
        let scratch = workspace
            .buffers(&self.workspace_requirements(SignalLayout::Contiguous(input.len()))?)?
            .values;
        inputs(input)?;
        self.run(input, output, scratch, execution)
    }
    /// Filters one validated record with `4 * n` scratch values, writing
    /// `output` only after both passes succeed.
    fn run<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = input.len();
        execution.check()?;
        if n == 0 {
            return Ok(());
        }
        let (forward, backward) = scratch[..4 * n].split_at_mut(2 * n);
        let at = |k: usize| input[if k < n { k } else { 2 * n - 1 - k }];
        let inverse = self.period_inverse(2 * n)?;
        let mut state = self.initial(2 * n, at, inverse, execution)?;
        for (k, y) in forward.iter_mut().enumerate() {
            if k.is_multiple_of(256) {
                execution.check()?;
            }
            *y = self.step(at(k), &mut state)?;
        }
        let mut state = self.initial(2 * n, |k| forward[2 * n - 1 - k], inverse, execution)?;
        for (k, y) in backward.iter_mut().rev().enumerate() {
            if k % 256 == 0 {
                execution.check()?;
            }
            *y = check(self.step(forward[2 * n - 1 - k], &mut state)? * T::from_real(self.gain))?;
        }
        execution.check()?;
        output[..n].copy_from_slice(&backward[..n]);
        Ok(())
    }
    fn step<T: NumericScalar<Real = S>>(
        &self,
        x: T,
        state: &mut [T; 2],
    ) -> Result<T, SciOxideError> {
        let y = check(x + T::from_real(self.a) * state[0] + T::from_real(self.b) * state[1])?;
        *state = [y, state[0]];
        Ok(y)
    }
    fn period_inverse(&self, n: usize) -> Result<[S; 4], SciOxideError> {
        // M^n for the two-component causal state; exponentiation avoids n-dependent setup.
        let mut power = [self.a, self.b, S::ONE, S::ZERO];
        let mut result = [S::ONE, S::ZERO, S::ZERO, S::ONE];
        let mul = |x: [S; 4], y: [S; 4]| {
            [
                x[0] * y[0] + x[1] * y[2],
                x[0] * y[1] + x[1] * y[3],
                x[2] * y[0] + x[3] * y[2],
                x[2] * y[1] + x[3] * y[3],
            ]
        };
        let mut exponent = n;
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = mul(result, power);
            }
            exponent /= 2;
            if exponent > 0 {
                power = mul(power, power);
            }
        }
        let m = [
            S::ONE - result[0],
            -result[1],
            -result[2],
            S::ONE - result[3],
        ];
        let det = finite(m[0] * m[3] - m[1] * m[2])?;
        if det == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "singular reflected filter initialization",
            ));
        }
        let inverse = [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det];
        for v in inverse {
            finite(v)?;
        }
        Ok(inverse)
    }
    fn initial<T: NumericScalar<Real = S>>(
        &self,
        n: usize,
        at: impl Fn(usize) -> T,
        inverse: [S; 4],
        execution: ExecutionContext<'_>,
    ) -> Result<[T; 2], SciOxideError> {
        let mut q = [T::ZERO; 2];
        for k in 0..n {
            if k % 256 == 0 {
                execution.check()?;
            }
            self.step(at(k), &mut q)?;
        }
        Ok([
            check(T::from_real(inverse[0]) * q[0] + T::from_real(inverse[1]) * q[1])?,
            check(T::from_real(inverse[2]) * q[0] + T::from_real(inverse[3]) * q[1])?,
        ])
    }
    /// Filters every lane along `axis` of an N-D array.
    ///
    /// Lanes run in parallel under the execution policy of the filter's options.
    ///
    /// # Errors
    ///
    /// The errors of [`SymmetricFilter::filter_into`] for each lane, and
    /// `InvalidInput` if `axis` is out of range or `DimensionMismatch` if the
    /// input and output shapes differ.
    pub fn filter_axis_into<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        axis_filter(
            input,
            output,
            axis,
            workspace,
            6,
            self.options.execution_policy(),
            execution,
            |x, y, w| {
                inputs(x)?;
                self.run(x, y, w, execution)
            },
        )
    }
}

/// Degree of the B-spline basis of a [`SplineFilter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplineDegree {
    /// Centered quadratic B-spline `β²`, with `β²(0) = 3/4` and `β²(±1) = 1/8`.
    Quadratic,
    /// Centered cubic B-spline `β³`, with `β³(0) = 2/3` and `β³(±1) = 1/6`.
    Cubic,
}
impl SplineDegree {
    fn value(self) -> usize {
        match self {
            Self::Quadratic => 2,
            Self::Cubic => 3,
        }
    }
}
/// Operation whose workspace size [`SplineFilter::workspace_requirements`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplineRequest<'a> {
    /// Computing the coefficients of a record or of every lane along an axis.
    Coefficients(SignalLayout<'a>),
    /// Reconstructing the samples of a record or of every lane along an axis.
    Reconstruct(SignalLayout<'a>),
    /// Evaluating the spline at the given number of points.
    Evaluate(usize),
}
/// B-spline interpolation of uniformly spaced samples by recursive filtering.
///
/// The interpolating spline of samples `x[0..n]` is
///
/// ```text
/// s(u) = Σₖ c[k] βᵖ(u − k),   s(i) = x[i] for i = 0, …, n − 1,
/// ```
///
/// where `βᵖ` is the centered B-spline of degree `p` and `u` is measured in
/// samples. The coefficients `c` extend by half-sample symmetric reflection,
/// which makes `s` even about `u = −1/2` and `u = n − 1/2`. They are the output
/// of the first-order [`SymmetricFilter`] with pole `−3 + 2√2` (quadratic)
/// or `−2 + √3` (cubic) and cost `O(n)`. Reconstruction at the samples and
/// evaluation at arbitrary points each cost `O(1)` per point. The filter holds
/// the degree and its recursive filter and computes nothing at construction.
///
/// ```
/// use scioxide_signal::{SignalExecutionOptions, SplineDegree, SplineFilter};
/// let spline = SplineFilter::<f64>::new(SplineDegree::Cubic, SignalExecutionOptions::new())?;
/// let samples = [1.0_f64, 2.0, 0.5, 3.0];
/// let coefficients = spline.coefficients(&samples)?;
/// let nodes = spline.reconstruct(&coefficients)?;
/// // The spline interpolates the samples.
/// assert!(
///     nodes
///         .iter()
///         .zip(samples)
///         .all(|(a, b)| (a - b).abs() < 1e-12)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Unser, A. Aldroubi and M. Eden, "B-spline signal processing: Part I—Theory"
/// and "Part II—Efficient design and applications", *IEEE Trans. Signal
/// Process.* 41(2), 821–848, 1993.
///
/// M. Unser, "Splines: a perfect fit for signal and image processing", *IEEE
/// Signal Process. Mag.* 16(6), 22–38, 1999.
#[derive(Clone, Debug)]
pub struct SplineFilter<S: RealScalar> {
    degree: SplineDegree,
    recursive: SymmetricFilter<S>,
}
impl<S: RealScalar> SplineFilter<S> {
    /// Prepares the interpolation filter of the given degree.
    ///
    /// The execution policy of `options` applies to the axis methods.
    ///
    /// # Errors
    ///
    /// Fails only if a filter constant cannot be represented in the scalar type.
    pub fn new(
        degree: SplineDegree,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let pole = match degree {
            SplineDegree::Quadratic => -scalar::<S>(3)? + (S::TWO) * (S::TWO).sqrt(),
            SplineDegree::Cubic => -(S::TWO) + scalar::<S>(3)?.sqrt(),
        };
        let neighbor = scioxide_interp::cardinal_bspline(degree.value(), S::ONE)?;
        Ok(Self {
            degree,
            recursive: SymmetricFilter::first_order(pole, -pole / neighbor, options)?,
        })
    }
    /// Returns the workspace size of an operation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the axis of an axis layout is out of range.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: SplineRequest<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        match request {
            SplineRequest::Coefficients(layout) => self.recursive.workspace_requirements(layout),
            SplineRequest::Reconstruct(SignalLayout::Contiguous(n))
            | SplineRequest::Evaluate(n) => Ok(SignalRequirements::values(n)),
            SplineRequest::Reconstruct(layout) => lane_requirements(layout, 2),
        }
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// The errors of [`SplineFilter::workspace_requirements`], and
    /// `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
        request: SplineRequest<'_>,
    ) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Reconstructs and returns the samples `s(i)` from the coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`SplineFilter::reconstruct_into`], and
    /// `AllocationFailure` if the result or workspace cannot be allocated.
    pub fn reconstruct<T: NumericScalar<Real = S>>(
        &self,
        coefficients: &[T],
    ) -> Result<Vec<T>, SciOxideError> {
        let mut output = zeros(coefficients.len())?;
        let mut workspace = self.workspace(SplineRequest::Reconstruct(
            SignalLayout::Contiguous(coefficients.len()),
        ))?;
        self.reconstruct_into(
            coefficients,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes and returns the spline coefficients of `input`.
    ///
    /// # Errors
    ///
    /// The errors of [`SplineFilter::coefficients_into`], and
    /// `AllocationFailure` if the result or workspace cannot be allocated.
    pub fn coefficients<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
    ) -> Result<Vec<T>, SciOxideError> {
        self.recursive.filter(input)
    }
    /// Computes the spline coefficients of `input` into the first `input.len()`
    /// entries of `output`.
    ///
    /// # Errors
    ///
    /// The errors of [`SymmetricFilter::filter_into`].
    pub fn coefficients_into<T: NumericScalar<Real = S>>(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.recursive
            .filter_into(input, output, workspace, execution)
    }
    /// Reconstructs the samples `s(i) = βᵖ(0) c[i] + βᵖ(1) (c[i − 1] + c[i + 1])`.
    ///
    /// Weighted terms use the product-sum accuracy described in the
    /// [crate docs](crate#accuracy), with overflow recovery before rounding.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is shorter than `coefficients` or the
    ///   workspace is too small.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a value overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn reconstruct_into<T: NumericScalar<Real = S>>(
        &self,
        coefficients: &[T],
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = coefficients.len();
        capacity(n, output.len())?;
        let staged = workspace.buffers(&SignalRequirements::values(n))?.values;
        inputs(coefficients)?;
        execution.check()?;
        self.reconstruct_record(coefficients, staged)?;
        execution.check()?;
        output[..n].copy_from_slice(staged);
        Ok(())
    }
    /// Reconstructs validated coefficients into `output` of the same length.
    fn reconstruct_record<T: NumericScalar<Real = S>>(
        &self,
        coefficients: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        let n = coefficients.len();
        let center = scioxide_interp::cardinal_bspline(self.degree.value(), S::ZERO)?;
        let neighbor = scioxide_interp::cardinal_bspline(self.degree.value(), S::ONE)?;
        for (i, y) in output[..n].iter_mut().enumerate() {
            *y = crate::util::working_sum_products(
                [
                    (coefficients[i], T::from_real(center)),
                    (coefficients[i.saturating_sub(1)], T::from_real(neighbor)),
                    (coefficients[(i + 1).min(n - 1)], T::from_real(neighbor)),
                ]
                .into_iter(),
            )?;
        }
        Ok(())
    }

    /// Evaluates the spline `s(u)` at `u = (x − origin) / spacing` for each
    /// coordinate `x`.
    ///
    /// Sample `i` lies at `x = origin + i · spacing`. Coordinates outside the
    /// record use the symmetric extension of `s`, which has period `2n`. The
    /// workspace must meet `SplineRequest::Evaluate(coordinates.len())`. Empty
    /// queries are valid. Coordinate differences retain extended exponents until
    /// division by `spacing`; periodic reduction preserves positive remainders.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `spacing` is not positive, or `coefficients` is empty
    ///   while `coordinates` is not.
    /// - `DimensionMismatch` if `output` is shorter than `coordinates` or the
    ///   workspace is too small.
    /// - `NonFinite` if a coefficient, coordinate, `origin` or `spacing` is NaN
    ///   or infinite.
    /// - `NumericalInstability` if `(x − origin) / spacing` or a value overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_into<T: NumericScalar<Real = S>>(
        &self,
        coefficients: &[T],
        coordinates: &[S],
        origin: S,
        spacing: S,
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::util::check_real_slice_finite(&[origin, spacing])?;
        if spacing <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("spacing"),
                "grid spacing must be positive",
            ));
        }
        capacity(coordinates.len(), output.len())?;
        let staged = workspace
            .buffers(&SignalRequirements::values(coordinates.len()))?
            .values;
        inputs(coefficients)?;
        inputs(coordinates)?;
        let n = coefficients.len();
        if n == 0 && !coordinates.is_empty() {
            return Err(empty(
                scioxide_core::argument!("coefficients"),
                "nonempty queries need at least one coefficient",
            ));
        }
        execution.check()?;
        if coordinates.is_empty() {
            return Ok(());
        }
        let period = scalar::<S>(size(n, 2)?)?;
        for (&x, y) in coordinates.iter().zip(staged.iter_mut()) {
            let u = scioxide_core::Scaled::from_value(x)?
                .sub(scioxide_core::Scaled::from_value(origin)?)?
                .div(scioxide_core::Scaled::from_value(spacing)?)?
                .round()?;
            let remainder = u % period;
            let u = if remainder < S::ZERO {
                remainder + period
            } else {
                remainder
            };
            let base = u.floor();
            // Reduction precedes conversion; all neighbor indices are small.
            let i = base.into_f64() as usize;
            let fraction = u - base;
            let mut pairs = [(T::ZERO, T::ZERO); 4];
            for (slot, offset) in pairs.iter_mut().zip(-1isize..=2) {
                let distance = fraction - S::try_from_f64_lossy(offset as f64)?;
                let basis = scioxide_interp::cardinal_bspline(self.degree.value(), distance)?;
                let index = if offset < 0 {
                    (i + 2 * n - 1) % (2 * n)
                } else {
                    (i + offset as usize) % (2 * n)
                };
                let index = if index < n { index } else { 2 * n - 1 - index };
                *slot = (coefficients[index], T::from_real(basis));
            }
            *y = crate::util::working_sum_products(pairs.into_iter())?;
        }
        execution.check()?;
        output[..coordinates.len()].copy_from_slice(staged);
        Ok(())
    }
    /// Computes the spline coefficients of every lane along `axis` of an N-D array.
    ///
    /// # Errors
    ///
    /// The errors of [`SymmetricFilter::filter_axis_into`].
    pub fn coefficients_axis_into<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.recursive
            .filter_axis_into(input, axis, output, workspace, execution)
    }
    /// Reconstructs the samples of every lane along `axis` of an N-D array.
    ///
    /// # Errors
    ///
    /// The errors of [`SplineFilter::reconstruct_into`] for each lane, and
    /// `InvalidInput` if `axis` is out of range or `DimensionMismatch` if the
    /// input and output shapes differ.
    pub fn reconstruct_axis_into<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        axis_filter(
            input,
            output,
            axis,
            workspace,
            2,
            self.recursive.options.execution_policy(),
            execution,
            |x, y, _| {
                inputs(x)?;
                self.reconstruct_record(x, y)
            },
        )
    }
}
