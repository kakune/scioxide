//! Evaluation plans for sparse polynomials and conversion to and from dense
//! coefficient tensors.
use super::{
    AxisPowers, Exact, ExactSparsePolynomial, axis_powers, power, validate_approx_terms,
    validate_borrowed, validate_permutation,
};
use crate::ExactCoefficient;
use crate::execution::dispatch;
use crate::util::{invalid, minimum_len};
use crate::util::{round_complex, scaled_complex};
use crate::{ApproxSparsePolynomial, MonomialOrder};
use crate::{PolynomialExecutionOptions, PolynomialScratchRequirements};
use alloc::borrow::Cow;
#[cfg(test)]
use core::num::NonZeroUsize;
use scioxide_core::ScaledValue;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{ExecutionContext, NdView, NdViewMut, NumericScalar, SciOxideError};

/// Evaluates a sparse polynomial with floating-point coefficients at one point
/// or at many points.
///
/// The plan borrows the exponent rows and coefficients, either of an
/// [`ApproxSparsePolynomial`](crate::ApproxSparsePolynomial) through its
/// `evaluation_plan` method or of sorted slices you provide through
/// [`Self::new`]. The method and accuracy are those of
/// [`ApproxSparsePolynomial::evaluate`](crate::ApproxSparsePolynomial::evaluate).
/// The plan precomputes the layout of the power table, the number of binary
/// powers `xᵢ^(2ᵏ)` each variable needs for its largest exponent, in O(t n)
/// time for `t` terms in `n` variables; a polynomial's plan borrows the layout
/// the polynomial already holds. [`Self::evaluate_batch_into`] evaluates the
/// rows of a matrix of points, in parallel when the execution options allow it.
///
/// # Examples
///
/// ```
/// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
///
/// // p(x, y) = 1 + 2x.
/// let order = MonomialOrder::GradedLexicographic;
/// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
/// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
/// let plan = p.evaluation_plan(Default::default())?;
/// let mut scratch = vec![0.0_f64; plan.workspace_requirements(1)?.scratch_len()];
/// assert!((plan.evaluate(&[3.0_f64, 4.0], &mut scratch)? - 7.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SparseEvaluationPlan<'a, C: NumericScalar> {
    dimension: usize,
    exponents: &'a [usize],
    coefficients: &'a [C],
    options: PolynomialExecutionOptions,
    axes: Cow<'a, [AxisPowers]>,
    lane_len: usize,
    native_coefficients: bool,
}
impl<'a, C: NumericScalar> SparseEvaluationPlan<'a, C> {
    /// Builds a plan that borrows flat exponent rows and coefficients.
    ///
    /// Term `j` has exponents `exponents[j·dimension..(j + 1)·dimension]` and
    /// coefficient `coefficients[j]`. The terms must already be in strictly
    /// ascending `order`, with no zero coefficient; unlike
    /// [`ApproxSparsePolynomial::new`](crate::ApproxSparsePolynomial::new), this
    /// constructor does not sort or combine terms. It allocates the table layout
    /// of the powers once.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `exponents.len()` is not
    ///   `dimension · coefficients.len()`.
    /// - `InvalidInput` if a coefficient is zero, the terms are not strictly
    ///   ascending, or the worker bound in `options` exceeds 64.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder, SparseEvaluationPlan};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let (exponents, coefficients) = (p.exponents(), p.coefficients());
    /// let plan = SparseEvaluationPlan::new(2, order, exponents, coefficients, Default::default())?;
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(1)?.scratch_len()];
    /// assert!((plan.evaluate(&[3.0_f64, 4.0], &mut scratch)? - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        dimension: usize,
        order: MonomialOrder,
        exponents: &'a [usize],
        coefficients: &'a [C],
        options: PolynomialExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        validate_approx_terms(dimension, order, exponents, coefficients)?;
        let axes = axis_powers(dimension, exponents, coefficients.len())?;
        Ok(Self::from_canonical(
            dimension,
            exponents,
            coefficients,
            options,
            Cow::Owned(axes),
            coefficients
                .iter()
                .copied()
                .all(crate::admission::native_value),
        ))
    }
    fn from_canonical(
        dimension: usize,
        exponents: &'a [usize],
        coefficients: &'a [C],
        options: PolynomialExecutionOptions,
        axes: Cow<'a, [AxisPowers]>,
        native_coefficients: bool,
    ) -> Self {
        let lane_len = axes.last().map_or(0, |axis| axis.offset + axis.bits);
        Self {
            dimension,
            exponents,
            coefficients,
            options,
            axes,
            lane_len,
            native_coefficients,
        }
    }
    /// Returns the execution options for batch evaluation.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder, PolynomialExecutionOptions};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let plan = p.evaluation_plan(Default::default())?;
    /// assert_eq!(plan.options(), PolynomialExecutionOptions::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn options(&self) -> PolynomialExecutionOptions {
        self.options
    }
    /// Returns `p(x)` at `point`.
    ///
    /// `scratch` needs `workspace_requirements(1)` entries. The method is that of
    /// [`ApproxSparsePolynomial::evaluate`](crate::ApproxSparsePolynomial::evaluate).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len()` is not the number of variables,
    ///   or `scratch` is too short.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `NumericalInstability` if the value overflows.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let plan = p.evaluation_plan(Default::default())?;
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(1)?.scratch_len()];
    /// assert!((plan.evaluate(&[3.0_f64, 4.0], &mut scratch)? - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: &[C], scratch: &mut [C]) -> Result<C, SciOxideError> {
        if point.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                self.dimension,
                point.len(),
            ));
        }
        minimum_len(self.workspace_requirements(1)?.scratch_len(), scratch.len())?;
        for &x in point {
            crate::util::finite_input(x, "nonfinite value")?;
        }
        let (stage, work) = scratch.split_at_mut(1);
        stage[0] = self.evaluate_with(|axis| Ok(point[axis]), work)?;
        Ok(stage[0])
    }
    fn evaluate_with(
        &self,
        point: impl Fn(usize) -> Result<C, SciOxideError>,
        scratch: &mut [C],
    ) -> Result<C, SciOxideError> {
        if let Some(value) = self.evaluate_native(&point, scratch)? {
            return Ok(value);
        }
        self.evaluate_extended(point)
    }
    fn evaluate_native(
        &self,
        point: &impl Fn(usize) -> Result<C, SciOxideError>,
        scratch: &mut [C],
    ) -> Result<Option<C>, SciOxideError> {
        // Coefficients are immutable: their native-range admission is cached at
        // preparation. Only point-dependent powers, products and sums need replay.
        if !self.native_coefficients {
            return Ok(None);
        }
        let safe = crate::admission::native_value;
        let axes = self.axes.as_ref();
        for (axis, layout) in axes.iter().enumerate() {
            if layout.bits == 0 {
                continue;
            }
            let table = &mut scratch[layout.offset..layout.offset + layout.bits];
            let mut x = point(axis)?;
            for (bit, slot) in table.iter_mut().enumerate() {
                if !safe(x) {
                    return Ok(None);
                }
                *slot = x;
                if bit + 1 < layout.bits {
                    x = x * x;
                }
            }
        }
        // Canonical zero-variable input has at most one coefficient. Keep the
        // same addition (including signed-zero conventions) as a normal reduction.
        if self.dimension == 0 {
            let coefficient = self.coefficients.first().copied().unwrap_or(C::ZERO);
            let value = C::ZERO + coefficient;
            return Ok((safe(coefficient) && safe(value)).then_some(value));
        }
        let mut sum = C::ZERO;
        for (exponents, &coefficient) in
            self.exponents.chunks(self.dimension).zip(self.coefficients)
        {
            let mut value = coefficient;
            for (&exponent, layout) in exponents.iter().zip(axes) {
                let mut remaining = exponent;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    value = value * scratch[layout.offset + bit];
                    if !safe(value) {
                        return Ok(None);
                    }
                    remaining &= remaining - 1;
                }
            }
            sum = sum + value;
            if !safe(sum) {
                return Ok(None);
            }
        }
        Ok(Some(sum))
    }
    fn evaluate_extended(
        &self,
        point: impl Fn(usize) -> Result<C, SciOxideError>,
    ) -> Result<C, SciOxideError> {
        // Preserve the original coefficient-first binary products and canonical
        // reduction on range fallback, including gradual final underflow.
        let mut sum = ScaledValue::ZERO;
        for (i, c) in self.coefficients.iter().enumerate() {
            let mut value = scaled_complex(*c)?;
            for (axis, &exponent) in self.exponents[i * self.dimension..(i + 1) * self.dimension]
                .iter()
                .enumerate()
            {
                let mut x = scaled_complex(point(axis)?)?;
                let mut n = exponent;
                while n != 0 {
                    if n & 1 != 0 {
                        value = value.mul(x)?;
                    }
                    n >>= 1;
                    if n != 0 {
                        x = x.mul(x)?;
                    }
                }
            }
            sum = sum.add(value)?;
        }
        round_complex(sum)
    }
    /// Returns the scratch length for evaluating `points` points.
    ///
    /// The scratch holds one result per point and, for each worker in use, a
    /// table of the powers `xᵢ^(2ᵏ)` with one entry per bit of the largest
    /// exponent of each variable. Pass `points = 1` for [`Self::evaluate`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, StridedLayout};
    /// use scioxide_poly::{MonomialOrder, PolynomialExecutionOptions, SparseEvaluationPlan};
    ///
    /// // p(x) = 2x at x = 1 and x = 3.
    /// let options = PolynomialExecutionOptions::new();
    /// let plan = SparseEvaluationPlan::new(1, MonomialOrder::Lexicographic, &[1], &[2.0], options)?;
    /// let (points, layout) = ([1.0_f64, 3.0], StridedLayout::new(&[2, 1], &[1, 1])?);
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(2)?.scratch_len()];
    /// let mut values = [0.0_f64; 2];
    /// let cx = ExecutionContext::sequential();
    /// plan.evaluate_batch_into(NdView::new(&points, layout)?, &mut values, &mut scratch, cx)?;
    /// assert!((values[0] - 2.0).abs() < 1e-13 && (values[1] - 6.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        let lanes = self.options.workers().get().min(points);
        Ok(PolynomialScratchRequirements::new(checked_add(
            points,
            checked_mul(
                lanes,
                self.lane_len,
                scioxide_core::quantity!("worker_scratch"),
            )?,
            scioxide_core::quantity!("batch_scratch"),
        )?))
    }
    /// Evaluates `p` at each row of `points` and writes the values to `output`.
    ///
    /// `points` has shape `[output.len(), n]`, one point per row. `scratch` needs
    /// `workspace_requirements(output.len())` entries. Rows are split among the
    /// workers that the plan's execution options and `context` allow, at most 64;
    /// each row gives the same result as [`Self::evaluate`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `points` does not have shape `[output.len(), n]`.
    /// - `DimensionMismatch` if `scratch` is too short.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `NumericalInstability` if a value overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, StridedLayout};
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let plan = p.evaluation_plan(Default::default())?;
    /// let (points, layout) = ([3.0_f64, 4.0], StridedLayout::new(&[1, 2], &[2, 1])?);
    /// let mut out = [0.0_f64];
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(1)?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// plan.evaluate_batch_into(NdView::new(&points, layout)?, &mut out, &mut scratch, cx)?;
    /// assert!((out[0] - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_into(
        &self,
        points: NdView<'_, '_, C>,
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let shape = points.layout().shape();
        if shape.extents() != [output.len(), self.dimension] {
            return Err(invalid(
                scioxide_core::argument!("points"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "batch shape must be [points, variables]",
            ));
        }
        for &x in points.iter() {
            crate::util::finite_input(x, "nonfinite value")?;
        }
        dispatch(
            output,
            scratch,
            self.lane_len,
            self.options.resolved_policy(),
            self.options.workers(),
            context,
            |row, work| self.evaluate_with(|axis| Ok(*points.get([row, axis])?), work),
        )
    }
}
impl<C: NumericScalar> ApproxSparsePolynomial<C> {
    /// Returns an evaluation plan that borrows this polynomial.
    ///
    /// The plan reuses the stored terms and power-table layout, so it is cheap
    /// to create.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the worker bound in `options` exceeds 64.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let plan = p.evaluation_plan(Default::default())?;
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(1)?.scratch_len()];
    /// assert!((plan.evaluate(&[3.0_f64, 4.0], &mut scratch)? - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluation_plan(
        &self,
        options: PolynomialExecutionOptions,
    ) -> Result<SparseEvaluationPlan<'_, C>, SciOxideError> {
        options.validate()?;
        Ok(SparseEvaluationPlan::from_canonical(
            self.dimension(),
            self.exponents(),
            self.coefficients(),
            options,
            Cow::Borrowed(self.axis_metadata()),
            self.native_coefficients(),
        ))
    }
    /// Builds a sparse polynomial from a dense tensor of monomial coefficients.
    ///
    /// The entry at tensor index `(i₀, …, iₙ₋₁)` is the coefficient of the
    /// monomial in which variable `axis_to_variable[a]` has exponent `iₐ`. Zero
    /// entries are skipped, and the terms are sorted in `order`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis_to_variable` is not a permutation of the tensor
    ///   axes.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the polynomial does not fit in
    ///   memory.
    ///
    /// ```
    /// use scioxide_core::{NdView, StridedLayout};
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // Row-major 2×2 tensor [[1, 0], [2, 0]] holds p(x, y) = 1 + 2x.
    /// let coefficients = [1.0_f64, 0.0, 2.0, 0.0];
    /// let tensor = NdView::new(&coefficients, StridedLayout::new(&[2, 2], &[2, 1])?)?;
    /// let order = MonomialOrder::Lexicographic;
    /// let p = ApproxSparsePolynomial::from_tensor(tensor, &[0, 1], order)?;
    /// assert!((p.evaluate(&[3.0_f64, 4.0])? - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_tensor(
        tensor: NdView<'_, '_, C>,
        axis_to_variable: &[usize],
        order: MonomialOrder,
    ) -> Result<Self, SciOxideError> {
        let shape = tensor.layout().shape();
        validate_permutation(
            axis_to_variable,
            shape.rank(),
            scioxide_core::argument!("axis_to_variable"),
        )?;
        let mut coefficients = scioxide_core::backend::try_with_capacity(
            shape.len(),
            scioxide_core::quantity!("sparse_terms"),
        )?;
        let mut exponents = scioxide_core::backend::try_with_capacity(
            checked_mul(
                shape.len(),
                shape.rank(),
                scioxide_core::quantity!("exponents"),
            )?,
            scioxide_core::quantity!("exponents"),
        )?;
        for (linear, &c) in tensor.iter().enumerate() {
            crate::util::finite_input(c, "nonfinite value")?;
            if c == C::ZERO {
                continue;
            }
            let mut index = linear;
            let start = exponents.len();
            exponents.resize(start + shape.rank(), 0);
            let e = &mut exponents[start..];
            // Decode the row-major linear index into per-axis indices (last axis fastest);
            // axis `a`'s index is the exponent of variable `axis_to_variable[a]`.
            for axis in (0..shape.rank()).rev() {
                let n = shape.extents()[axis];
                e[axis_to_variable[axis]] = index % n;
                index /= n;
            }
            coefficients.push(c);
        }
        Self::new(shape.rank(), order, exponents, coefficients)
    }
    /// Writes the coefficients to a dense tensor, with zeros for absent
    /// monomials.
    ///
    /// Tensor index `(i₀, …, iₙ₋₁)` receives the coefficient of the monomial in
    /// which variable `axis_to_variable[a]` has exponent `iₐ`. Every exponent is
    /// checked against the tensor extents before the tensor is written.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis_to_variable` is not a permutation of `0..n`.
    /// - `DimensionMismatch` if the tensor rank is not `n`, or an extent is too
    ///   small for an exponent.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdViewMut, StridedLayout};
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x, y) = 1 + 2x.
    /// let order = MonomialOrder::GradedLexicographic;
    /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
    /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
    /// let layout = StridedLayout::new(&[2, 2], &[2, 1])?;
    /// let mut coefficients = [0.0_f64; 4];
    /// let tensor = NdViewMut::new(&mut coefficients, layout)?;
    /// p.to_tensor_into(tensor, &[0, 1], ExecutionContext::sequential())?;
    /// assert_eq!(coefficients, [1.0, 0.0, 2.0, 0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_tensor_into(
        &self,
        mut tensor: NdViewMut<'_, '_, C>,
        axis_to_variable: &[usize],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        validate_permutation(
            axis_to_variable,
            self.dimension(),
            scioxide_core::argument!("axis_to_variable"),
        )?;
        let shape = tensor.layout().shape();
        if shape.rank() != self.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                [self.dimension()],
                [shape.rank()],
            ));
        }
        // The first unrepresentable exponent reports its axis' required extent.
        for (m, _) in self.terms() {
            for (exponent, &extent) in
                tensor_coordinates(axis_to_variable, m.exponents()).zip(shape.extents())
            {
                if exponent >= extent {
                    return Err(SciOxideError::dimension_mismatch([exponent + 1], [extent]));
                }
            }
        }
        context.check()?;
        tensor.for_each_mut(|c| *c = C::ZERO);
        for (m, c) in self.terms() {
            *tensor.get_mut(tensor_coordinates(axis_to_variable, m.exponents()))? = *c;
        }
        Ok(())
    }
}

/// Returns the tensor index of one term: axis `a` holds `exponents[axis_to_variable[a]]`.
fn tensor_coordinates<'a>(
    axis_to_variable: &'a [usize],
    exponents: &'a [usize],
) -> impl ExactSizeIterator<Item = usize> + 'a {
    axis_to_variable
        .iter()
        .map(move |&variable| exponents[variable])
}

/// Borrows the terms of a sparse polynomial with exact coefficients and
/// evaluates it at one point or at many points.
///
/// The view borrows the exponent rows and coefficients, either of an
/// [`ExactSparsePolynomial`](crate::ExactSparsePolynomial) through its
/// `as_view` method or of sorted slices you provide through [`Self::new`]. It
/// precomputes nothing. The method is that of
/// [`ExactSparsePolynomial::evaluate`](crate::ExactSparsePolynomial::evaluate).
/// Arithmetic in your coefficient type may allocate.
///
/// # Examples
///
/// ```
/// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
///
/// // p(x, y) = 1 + 2x over the rationals.
/// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
/// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
/// let view = p.as_view();
/// assert_eq!(view.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ExactSparseView<'a, C: ExactCoefficient> {
    dimension: usize,
    exponents: &'a [usize],
    coefficients: &'a [C],
}
impl<'a, C: ExactCoefficient> ExactSparseView<'a, C> {
    /// Builds a view that borrows flat exponent rows and coefficients, without
    /// allocating.
    ///
    /// Term `j` has exponents `exponents[j·dimension..(j + 1)·dimension]` and
    /// coefficient `coefficients[j]`. The terms must already be in strictly
    /// ascending `order`, with no zero coefficient.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `exponents.len()` is not
    ///   `dimension · coefficients.len()`.
    /// - `InvalidInput` if a coefficient is zero or the terms are not strictly
    ///   ascending.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, ExactSparseView, MonomialOrder};
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x over the rationals.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// let (exponents, coefficients) = (p.exponents(), p.coefficients());
    /// let view = ExactSparseView::new(2, p.order(), exponents, coefficients)?;
    /// assert_eq!(view.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        dimension: usize,
        order: MonomialOrder,
        exponents: &'a [usize],
        coefficients: &'a [C],
    ) -> Result<Self, SciOxideError> {
        validate_borrowed::<C, Exact>(dimension, order, exponents, coefficients)?;
        Ok(Self {
            dimension,
            exponents,
            coefficients,
        })
    }
    /// Returns `p(x)` at `point`, computed exactly.
    ///
    /// The terms are summed in ascending order, and each power `xᵢ^αᵢ` is formed
    /// by repeated squaring.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len()` is not the number of variables.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x over the rationals.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// let view = p.as_view();
    /// assert_eq!(view.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: &[C]) -> Result<C, SciOxideError> {
        if point.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                self.dimension,
                point.len(),
            ));
        }
        self.evaluate_with(|axis| Ok(&point[axis]))
    }
    fn evaluate_with<'p>(
        &self,
        point: impl Fn(usize) -> Result<&'p C, SciOxideError>,
    ) -> Result<C, SciOxideError>
    where
        C: 'p,
    {
        self.coefficients
            .iter()
            .enumerate()
            .try_fold(C::zero(), |sum, (i, c)| {
                let value = self.exponents[i * self.dimension..(i + 1) * self.dimension]
                    .iter()
                    .enumerate()
                    .try_fold(c.clone(), |value, (axis, &e)| {
                        Ok::<_, SciOxideError>(value * power::<C, Exact>(point(axis)?.clone(), e)?)
                    })?;
                Ok(sum + value)
            })
    }
    /// Returns the scratch length for evaluating `points` points: one
    /// coefficient per point.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the `workspace_requirements` methods of the
    /// other evaluation forms.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x over the rationals.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// let view = p.as_view();
    /// assert!(view.workspace_requirements(2)?.scratch_len() >= 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<crate::PolynomialScratchRequirements, SciOxideError> {
        Ok(crate::PolynomialScratchRequirements::new(points))
    }
    /// Evaluates `p` at each row of `points`, one row after another, and writes
    /// the values to `output`.
    ///
    /// `points` has shape `[output.len(), n]`, one point per row. `scratch` needs
    /// `workspace_requirements(output.len())` entries.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `points` does not have shape `[output.len(), n]`.
    /// - `DimensionMismatch` if `scratch` is too short.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, StridedLayout};
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x over the rationals.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// let view = p.as_view();
    /// let points = NdView::new(&[Q(3, 1), Q(4, 1)], StridedLayout::new(&[1, 2], &[2, 1])?)?;
    /// let mut output = [Q(0, 1)];
    /// let mut scratch = vec![Q(0, 1); view.workspace_requirements(1)?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// view.evaluate_batch_into(points, &mut output, &mut scratch, cx)?;
    /// assert_eq!(output[0], Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_into(
        &self,
        points: scioxide_core::NdView<'_, '_, C>,
        output: &mut [C],
        scratch: &mut [C],
        context: scioxide_core::ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if points.layout().shape().extents() != [output.len(), self.dimension] {
            return Err(invalid(
                scioxide_core::argument!("points"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "exact batch shape mismatch",
            ));
        }
        crate::util::minimum_len(output.len(), scratch.len())?;
        context.check()?;
        // Results are staged in `scratch` so cancellation or an error never leaves
        // `output` partially written.
        for (row, slot) in scratch[..output.len()].iter_mut().enumerate() {
            context.check()?;
            *slot = self.evaluate_with(|axis| points.get([row, axis]))?;
        }
        context.check()?;
        output.clone_from_slice(&scratch[..output.len()]);
        Ok(())
    }
}
impl<C: ExactCoefficient> ExactSparsePolynomial<C> {
    /// Returns a view that borrows this polynomial's terms.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x over the rationals.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// let view = p.as_view();
    /// assert_eq!(view.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_view(&self) -> ExactSparseView<'_, C> {
        ExactSparseView {
            dimension: self.dimension(),
            exponents: &self.data.exponents,
            coefficients: &self.data.coefficients,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_admission_is_rebuilt_after_folding_and_arithmetic() -> Result<(), SciOxideError>
    {
        let order = MonomialOrder::Lexicographic;
        let large = ApproxSparsePolynomial::new(1, order, alloc::vec![1], alloc::vec![1e200_f64])?;
        let small = ApproxSparsePolynomial::new(1, order, alloc::vec![0], alloc::vec![1e-200_f64])?;
        let scaled = large.try_mul(&small)?;
        let zero = large.try_sub(&large)?;
        let folded = ApproxSparsePolynomial::new(
            1,
            order,
            alloc::vec![1, 1, 1],
            alloc::vec![1e200_f64, -1e200, 1.],
        )?;
        let mut scratch = [0.; 4];
        assert!(
            large
                .evaluation_plan(Default::default())?
                .evaluate_native(&|_| Ok(1.), &mut scratch)?
                .is_none()
        );
        assert!(
            large
                .derivative(0)?
                .evaluation_plan(Default::default())?
                .evaluate_native(&|_| Ok(1.), &mut scratch)?
                .is_none()
        );
        for (p, expected) in [(&scaled, 1.), (&zero, 0.), (&folded, 1.)] {
            let native = p
                .evaluation_plan(Default::default())?
                .evaluate_native(&|_| Ok(1.), &mut scratch)?;
            assert!(native.is_some());
            if let Some(value) = native {
                assert!((value - expected).abs() <= 1e-14 * expected.abs().max(1.));
            }
        }
        Ok(())
    }

    #[test]
    fn mixed_scale_points_admit_only_required_axis_powers() -> Result<(), SciOxideError> {
        let p = ApproxSparsePolynomial::new(
            3,
            MonomialOrder::Lexicographic,
            alloc::vec![1, 0, 0, 0, 64, 0],
            alloc::vec![1.0_f64, 1.0],
        )?;
        let plan = p.evaluation_plan(Default::default())?;
        assert_eq!(plan.lane_len, 8);
        assert_eq!(plan.workspace_requirements(1)?.scratch_len(), 9);
        let mut scratch = [0.; 9];
        for point in [
            [1.5, 1.0, f64::MAX],
            [1e10, 1.0, 1e-300],
            [1e-40, 0.5, f64::MAX],
        ] {
            let expected = point[0] + (0..64).fold(1.0, |v, _| v * point[1]);
            let native = plan.evaluate_native(&|axis| Ok(point[axis]), &mut scratch)?;
            assert!(native.is_some());
            if let Some(value) = native {
                assert!((value / expected - 1.).abs() < 1e-14);
            }
            assert!((plan.evaluate(&point, &mut scratch)? / expected - 1.).abs() < 1e-14);
        }
        for (coefficient, point, expected) in [
            (1e-300, 1e200, 1e100),
            (1e300, 1e-200, 1e-100),
            (1., 1e-160, 1e-320),
        ] {
            let p = ApproxSparsePolynomial::new(
                1,
                MonomialOrder::Lexicographic,
                alloc::vec![2],
                alloc::vec![coefficient],
            )?;
            let plan = p.evaluation_plan(Default::default())?;
            assert!(
                plan.evaluate_native(&|_| Ok(point), &mut scratch)?
                    .is_none()
            );
            let value = plan.evaluate(&[point], &mut scratch)?;
            assert!((value - expected).abs() <= expected.abs() * 1e-14 + f64::from_bits(1));
        }
        Ok(())
    }

    #[test]
    fn excessive_batch_workers_preserve_output() {
        let mut output = [99.0_f64; 2];
        let error = dispatch(
            &mut output,
            &mut [0.; 2],
            0,
            scioxide_core::ExecutionPolicy::Sequential,
            NonZeroUsize::MIN.saturating_add(64),
            ExecutionContext::sequential(),
            |_, _| Ok(1.),
        )
        .err();
        assert!(matches!(
            error.as_ref().map(SciOxideError::kind),
            Some(scioxide_core::SciOxideErrorKind::InvalidInput)
        ));
        assert_eq!(output, [99.; 2]);
        let result = dispatch(
            &mut output,
            &mut [0.; 2],
            0,
            scioxide_core::ExecutionPolicy::Sequential,
            NonZeroUsize::MIN.saturating_add(63),
            ExecutionContext::sequential(),
            |_, _| Ok(1.),
        );
        assert!(result.is_ok());
        assert_eq!(output, [1.; 2]);
    }
}
