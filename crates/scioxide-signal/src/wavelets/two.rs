use super::axis::{axis_transform, axis_workspace};
use super::*;
use scioxide_core::{NdView, NdViewMut, StridedLayout};

/// Arrangement of the levels of a separable 2-D wavelet transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wavelet2dLayout {
    /// Standard (Mallat) pyramid: each level transforms both axes once and then
    /// decomposes only the low/low block. Both axes have the same depth.
    ///
    /// The packed coefficients hold the final low/low block, then for each level
    /// from coarsest to finest the blocks with detail levels `[0, l]`, `[l, 0]`
    /// and `[l, l]`.
    Pyramid,
    /// Tensor-product transform: a full multilevel 1-D transform along each axis,
    /// with independent depths.
    ///
    /// The packed coefficients hold the product of the 1-D bands of the two axes
    /// (final approximation, then details from coarsest to finest), with axis 0
    /// outermost.
    Tensor,
}
/// Rectangular block of a 2-D wavelet transform within the packed coefficients.
///
/// The block is stored contiguously in row-major order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveletSubband {
    offset: usize,
    shape: [usize; 2],
    detail_levels: [usize; 2],
}
impl WaveletSubband {
    /// Returns the offset of the block's first entry in the packed coefficients.
    pub const fn offset(self) -> usize {
        self.offset
    }
    /// Returns the block shape `[rows, columns]`.
    pub const fn shape(self) -> [usize; 2] {
        self.shape
    }
    /// Returns the detail level along axis 0 and axis 1.
    ///
    /// Level 1 is the finest. Zero means the block is lowpass along that axis.
    pub const fn detail_levels(self) -> [usize; 2] {
        self.detail_levels
    }
}
#[derive(Clone, Debug)]
struct Stage<T: NumericScalar> {
    row: WaveletPlan<T>,
    col: WaveletPlan<T>,
    shape: [usize; 2],
    coeff_shape: [usize; 2],
}
/// Separable 2-D discrete wavelet transform of a dense row-major matrix.
///
/// Each level applies the 1-D transform of [`WaveletPlan`] to the rows and the
/// columns. The results equal the composition of the 1-D transforms bit for bit,
/// under every execution policy. [`Wavelet2dLayout`] selects the pyramid or the
/// tensor-product arrangement, and [`Wavelet2dPlan::subbands`] describes the
/// packed blocks. One workspace serves both directions.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{Wavelet2dLayout, Wavelet2dPlan, WaveletFamily, WaveletOptions};
/// let plan = Wavelet2dPlan::<f64>::new(
///     WaveletFamily::Haar,
///     [2, 2],
///     [1, 1],
///     Wavelet2dLayout::Pyramid,
///     WaveletOptions::default(),
/// )?;
/// let mut ws = plan.workspace(())?;
/// let mut coefficients = [0.0; 4];
/// let cx = ExecutionContext::sequential_native();
/// plan.analyze_into(&[1.0; 4], &mut coefficients, &mut ws, cx)?;
/// // The low/low Haar coefficient of a 2×2 block of ones is 4 · (1/√2)² = 2.
/// assert!((coefficients[0] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. G. Mallat, "A theory for multiresolution signal decomposition: the wavelet
/// representation", *IEEE Trans. Pattern Anal. Mach. Intell.* 11(7), 674–693, 1989.
#[derive(Clone, Debug)]
pub struct Wavelet2dPlan<T: NumericScalar> {
    shape: [usize; 2],
    layout: Wavelet2dLayout,
    stages: Vec<Stage<T>>,
    bands: Vec<WaveletSubband>,
    coefficients: usize,
    width: usize,
    axis_work: usize,
}
fn volume(s: [usize; 2]) -> Result<usize, SciOxideError> {
    checked_mul(s[0], s[1], scioxide_core::quantity!("wavelet_matrix"))
}
fn axis_bands<T: NumericScalar>(
    p: &WaveletPlan<T>,
) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    core::iter::once((0, p.approximation, 0)).chain(
        p.levels
            .iter()
            .enumerate()
            .rev()
            .map(|(i, l)| (l.detail_offset, l.band_len, i + 1)),
    )
}
impl<T: NumericScalar> Wavelet2dPlan<T> {
    /// Prepares a transform of matrices with `shape = [rows, columns]`.
    ///
    /// `levels[0]` is the depth along axis 0 (over `shape[0]` rows) and
    /// `levels[1]` the depth along axis 1. Each depth is limited as in
    /// [`WaveletPlan::new`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the family order is not supported, a depth exceeds its
    ///   limit, or a pyramid layout has unequal depths.
    /// - `SizeOverflow` if the coefficient or workspace length overflows `usize`.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(
        family: WaveletFamily,
        shape: [usize; 2],
        levels: [usize; 2],
        layout: Wavelet2dLayout,
        options: WaveletOptions,
    ) -> Result<Self, SciOxideError> {
        if levels[0] > maximum_depth(shape[0], family, options.boundary)?
            || levels[1] > maximum_depth(shape[1], family, options.boundary)?
        {
            return Err(invalid(
                scioxide_core::argument!("levels"),
                "levels exceed the maximum decomposition depth",
            ));
        }
        if layout == Wavelet2dLayout::Pyramid && levels[0] != levels[1] {
            return Err(inconsistent(
                scioxide_core::argument!("levels"),
                "a pyramid layout requires equal levels on both axes",
            ));
        }
        let count = if layout == Wavelet2dLayout::Tensor {
            1
        } else {
            levels[0]
        };
        let mut stages = try_with_capacity(count, scioxide_core::quantity!("wavelet_2d_stages"))?;
        let mut current = shape;
        let mut width = volume(shape)?;
        let mut axis_work = 0;
        // Validate the family even for a zero-level pyramid.
        if count == 0 {
            WaveletFilters::<T::Real>::new(family)?;
        }
        for _ in 0..count {
            let depth = if layout == Wavelet2dLayout::Tensor {
                levels
            } else {
                [1, 1]
            };
            let extent = scioxide_core::argument!("shape");
            let row = WaveletPlan::prepare(family, current[1], depth[1], options, extent)?;
            let col = WaveletPlan::prepare(family, current[0], depth[0], options, extent)?;
            let coeff_shape = [col.coefficients, row.coefficients];
            width = width
                .max(volume(current)?)
                .max(volume(coeff_shape)?)
                .max(checked_mul(
                    current[0],
                    row.coefficients,
                    scioxide_core::quantity!("wavelet_intermediate"),
                )?);
            axis_work = axis_work
                .max(axis_workspace(&row, current[0])?)
                .max(axis_workspace(&col, row.coefficients)?);
            let next = [col.approximation, row.approximation];
            stages.push(Stage {
                row,
                col,
                shape: current,
                coeff_shape,
            });
            current = next;
        }
        let mut bands = try_with_capacity(
            if layout == Wavelet2dLayout::Tensor {
                checked_mul(
                    levels[0] + 1,
                    levels[1] + 1,
                    scioxide_core::quantity!("wavelet_bands"),
                )?
            } else {
                checked_add(
                    checked_mul(count, 3, scioxide_core::quantity!("wavelet_bands"))?,
                    1,
                    scioxide_core::quantity!("wavelet_bands"),
                )?
            },
            scioxide_core::quantity!("wavelet_bands"),
        )?;
        let mut coefficients = 0;
        if layout == Wavelet2dLayout::Tensor {
            let s = &stages[0];
            for (_, nr, lr) in axis_bands(&s.col) {
                for (_, nc, lc) in axis_bands(&s.row) {
                    bands.push(WaveletSubband {
                        offset: coefficients,
                        shape: [nr, nc],
                        detail_levels: [lr, lc],
                    });
                    coefficients = checked_add(
                        coefficients,
                        checked_mul(nr, nc, scioxide_core::quantity!("wavelet_band"))?,
                        scioxide_core::quantity!("wavelet_packing"),
                    )?;
                }
            }
        } else {
            coefficients = volume(current)?;
            bands.push(WaveletSubband {
                offset: 0,
                shape: current,
                detail_levels: [0, 0],
            });
            for (i, s) in stages.iter().enumerate().rev() {
                let ar = s.col.approximation;
                let ac = s.row.approximation;
                for (shape, detail_levels) in [
                    ([ar, ac], [0, i + 1]),
                    ([ar, ac], [i + 1, 0]),
                    ([ar, ac], [i + 1, i + 1]),
                ] {
                    bands.push(WaveletSubband {
                        offset: coefficients,
                        shape,
                        detail_levels,
                    });
                    coefficients = checked_add(
                        coefficients,
                        volume(shape)?,
                        scioxide_core::quantity!("wavelet_packing"),
                    )?;
                }
            }
        }
        let result = Self {
            shape,
            layout,
            stages,
            bands,
            coefficients,
            width,
            axis_work,
        };
        // Reject unrepresentable workspace sizes during preparation.
        let _ = result.workspace_requirements(())?;
        Ok(result)
    }
    /// Returns the matrix shape `[rows, columns]`.
    pub fn input_shape(&self) -> [usize; 2] {
        self.shape
    }
    /// Returns the layout.
    pub fn layout(&self) -> Wavelet2dLayout {
        self.layout
    }
    /// Returns the blocks of the packed coefficients in storage order.
    pub fn subbands(&self) -> &[WaveletSubband] {
        &self.bands
    }
    /// Returns the number of packed coefficients.
    pub fn coefficient_len(&self) -> usize {
        self.coefficients
    }
    /// Returns the workspace size for analysis and synthesis.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        _request: (),
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(checked_add(
            checked_add(
                self.coefficients.max(volume(self.shape)?),
                checked_mul(
                    3,
                    self.width,
                    scioxide_core::quantity!("wavelet_2d_scratch"),
                )?,
                scioxide_core::quantity!("wavelet_2d_scratch"),
            )?,
            self.axis_work,
            scioxide_core::quantity!("wavelet_2d_scratch"),
        )?))
    }
    /// Allocates a workspace for analysis and synthesis.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        s: &Stage<T>,
        input: &[T],
        output: &mut [T],
        intermediate: &mut [T],
        work: &mut [T],
        inverse: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let middle = [s.shape[0], s.coeff_shape[1]];
        if matches!(s.row.options.execution_policy, ExecutionPolicy::Sequential) {
            if inverse {
                dense_axis(
                    &s.col,
                    input,
                    intermediate,
                    middle[1],
                    false,
                    work,
                    true,
                    context,
                )?;
                dense_axis(
                    &s.row,
                    &intermediate[..middle[0] * middle[1]],
                    output,
                    middle[1],
                    true,
                    work,
                    true,
                    context,
                )?;
            } else {
                dense_axis(
                    &s.row,
                    &input[..s.shape[0] * s.shape[1]],
                    intermediate,
                    s.shape[1],
                    true,
                    work,
                    false,
                    context,
                )?;
                dense_axis(
                    &s.col,
                    &intermediate[..middle[0] * middle[1]],
                    output,
                    middle[1],
                    false,
                    work,
                    false,
                    context,
                )?;
            }
            return Ok(());
        }
        if inverse {
            axis_transform(
                &s.col,
                NdView::new(
                    input,
                    StridedLayout::new(&s.coeff_shape, &[s.coeff_shape[1], 1])?,
                )?,
                NdViewMut::new(intermediate, StridedLayout::new(&middle, &[middle[1], 1])?)?,
                0,
                work,
                true,
                context,
            )?;
            axis_transform(
                &s.row,
                NdView::new(intermediate, StridedLayout::new(&middle, &[middle[1], 1])?)?,
                NdViewMut::new(output, StridedLayout::new(&s.shape, &[s.shape[1], 1])?)?,
                1,
                work,
                true,
                context,
            )
        } else {
            axis_transform(
                &s.row,
                NdView::new(input, StridedLayout::new(&s.shape, &[s.shape[1], 1])?)?,
                NdViewMut::new(intermediate, StridedLayout::new(&middle, &[middle[1], 1])?)?,
                1,
                work,
                false,
                context,
            )?;
            axis_transform(
                &s.col,
                NdView::new(intermediate, StridedLayout::new(&middle, &[middle[1], 1])?)?,
                NdViewMut::new(
                    output,
                    StridedLayout::new(&s.coeff_shape, &[s.coeff_shape[1], 1])?,
                )?,
                0,
                work,
                false,
                context,
            )
        }
    }
    fn raw(
        &self,
        input: &[T],
        output: &mut [T],
        work: &mut [T],
        inverse: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (a, rest) = work.split_at_mut(self.width);
        let (b, rest) = rest.split_at_mut(self.width);
        let (middle, axis_work) = rest.split_at_mut(self.width);
        if self.layout == Wavelet2dLayout::Tensor {
            let s = &self.stages[0];
            if inverse {
                self.tensor_pack(s, input, a, true);
                self.pass(s, a, output, middle, axis_work, true, context)?;
            } else {
                self.pass(s, input, a, middle, axis_work, false, context)?;
                self.tensor_pack(s, a, output, false);
            }
            return Ok(());
        }
        if inverse {
            let n = volume(self.bands[0].shape)?;
            a[..n].copy_from_slice(&input[..n]);
            for (index, s) in self.stages.iter().enumerate().rev() {
                context.check()?;
                let nr = s.col.approximation;
                let nc = s.row.approximation;
                copy_rect(a, [nr, nc], [0, 0], b, s.coeff_shape, [0, 0], [nr, nc]);
                let first = 1 + 3 * (self.stages.len() - 1 - index);
                for (band, start) in
                    self.bands[first..first + 3]
                        .iter()
                        .zip([[0, nc], [nr, 0], [nr, nc]])
                {
                    copy_rect(
                        &input[band.offset..],
                        band.shape,
                        [0, 0],
                        b,
                        s.coeff_shape,
                        start,
                        band.shape,
                    );
                }
                self.pass(s, b, a, middle, axis_work, true, context)?;
            }
            output.copy_from_slice(&a[..volume(self.shape)?]);
        } else {
            a[..input.len()].copy_from_slice(input);
            for (index, s) in self.stages.iter().enumerate() {
                context.check()?;
                self.pass(s, a, b, middle, axis_work, false, context)?;
                let nr = s.col.approximation;
                let nc = s.row.approximation;
                let first = 1 + 3 * (self.stages.len() - 1 - index);
                for (band, start) in
                    self.bands[first..first + 3]
                        .iter()
                        .zip([[0, nc], [nr, 0], [nr, nc]])
                {
                    copy_rect(
                        b,
                        s.coeff_shape,
                        start,
                        &mut output[band.offset..],
                        band.shape,
                        [0, 0],
                        band.shape,
                    );
                }
                copy_rect(b, s.coeff_shape, [0, 0], a, [nr, nc], [0, 0], [nr, nc]);
            }
            let n = volume(self.bands[0].shape)?;
            output[..n].copy_from_slice(&a[..n]);
        }
        Ok(())
    }
    fn tensor_pack(&self, s: &Stage<T>, input: &[T], output: &mut [T], inverse: bool) {
        let mut bands = self.bands.iter();
        for (r, nr, _) in axis_bands(&s.col) {
            for (c, nc, _) in axis_bands(&s.row) {
                if let Some(band) = bands.next() {
                    if inverse {
                        copy_rect(
                            &input[band.offset..],
                            [nr, nc],
                            [0, 0],
                            output,
                            s.coeff_shape,
                            [r, c],
                            [nr, nc],
                        );
                    } else {
                        copy_rect(
                            input,
                            s.coeff_shape,
                            [r, c],
                            &mut output[band.offset..],
                            [nr, nc],
                            [0, 0],
                            [nr, nc],
                        );
                    }
                }
            }
        }
    }
    /// Validates lengths and finiteness for one direction, then stages the
    /// transform in the workspace and copies it to the output after success.
    fn transform(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        inverse: bool,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = volume(self.shape)?;
        let (a, b) = if inverse {
            (self.coefficients, n)
        } else {
            (n, self.coefficients)
        };
        exact_len(a, input.len())?;
        exact_len(b, output.len())?;
        let buffers = ws.buffers(&self.workspace_requirements(())?)?;
        finite(input)?;
        cx.check()?;
        let (stage, scratch) = buffers.values.split_at_mut(n.max(self.coefficients));
        self.raw(input, &mut stage[..b], scratch, inverse, cx)?;
        cx.check()?;
        output.copy_from_slice(&stage[..b]);
        Ok(())
    }
    /// Computes the packed wavelet coefficients of a row-major matrix.
    ///
    /// `output` holds [`Wavelet2dPlan::coefficient_len`] values.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `rows · columns` entries,
    ///   `output` does not have [`Wavelet2dPlan::coefficient_len`] entries, or the
    ///   workspace is too small.
    /// - `NonFinite` if an input sample is NaN or infinite.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn analyze_into(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform(input, output, ws, false, cx)
    }
    /// Reconstructs a row-major matrix from packed coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`Wavelet2dPlan::analyze_into`], with the roles of the two
    /// lengths exchanged.
    pub fn synthesize_into(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.transform(input, output, ws, true, cx)
    }
    /// Computes and returns the packed wavelet coefficients of a row-major matrix.
    ///
    /// # Errors
    ///
    /// The errors of [`Wavelet2dPlan::analyze_into`], and `AllocationFailure` if
    /// the result or workspace cannot be allocated.
    pub fn analyze(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        self.allocate(input, false)
    }
    /// Reconstructs and returns a row-major matrix from packed coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`Wavelet2dPlan::synthesize_into`], and `AllocationFailure`
    /// if the result or workspace cannot be allocated.
    pub fn synthesize(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        self.allocate(input, true)
    }
    fn allocate(&self, input: &[T], inverse: bool) -> Result<Vec<T>, SciOxideError> {
        let mut out = try_filled(
            if inverse {
                volume(self.shape)?
            } else {
                self.coefficients
            },
            T::ZERO,
            scioxide_core::quantity!("wavelet_2d_output"),
        )?;
        let mut ws = self.workspace(())?;
        self.transform(
            input,
            &mut out,
            &mut ws,
            inverse,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
}
fn copy_rect<T: Copy>(
    input: &[T],
    input_shape: [usize; 2],
    start: [usize; 2],
    output: &mut [T],
    output_shape: [usize; 2],
    dest: [usize; 2],
    shape: [usize; 2],
) {
    for r in 0..shape[0] {
        let a = (r + start[0]) * input_shape[1] + start[1];
        let b = (r + dest[0]) * output_shape[1] + dest[1];
        output[b..b + shape[1]].copy_from_slice(&input[a..a + shape[1]]);
    }
}

/// Dense internal stages receive validated input and staged output storage.
#[allow(clippy::too_many_arguments)]
fn dense_axis<T: NumericScalar>(
    plan: &WaveletPlan<T>,
    input: &[T],
    output: &mut [T],
    columns: usize,
    rows: bool,
    work: &mut [T],
    inverse: bool,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let (n, m) = if inverse {
        (plan.coefficients, plan.n)
    } else {
        (plan.n, plan.coefficients)
    };
    if n == 0 || m == 0 || columns == 0 {
        return Ok(());
    }
    if rows {
        for (x, y) in input.chunks_exact(n).zip(output.chunks_exact_mut(m)) {
            context.check()?;
            plan.raw(x, y, work, inverse, context)?;
        }
    } else {
        if !inverse && column_analysis(plan, input, output, columns, work, context)? {
            return Ok(());
        }
        let (x, rest) = work.split_at_mut(n);
        let (y, scratch) = rest.split_at_mut(m);
        for column in 0..columns {
            context.check()?;
            for (row, slot) in x.iter_mut().enumerate() {
                *slot = input[row * columns + column];
            }
            plan.raw(x, y, scratch, inverse, context)?;
            for (row, &value) in y.iter().enumerate() {
                output[row * columns + column] = value;
            }
        }
    }
    Ok(())
}

/// Transform column lanes together so each filter row reads contiguous memory.
/// Every coefficient uses the 1-D periodic row order `(x0*h0 + x1*h1) +
/// (x2*h2 + x3*h3)`, wrapped or not, so lanes equal the per-column kernel
/// bitwise. The matrix caller stages the output; exceptional ranges
/// recompute lanes.
fn column_analysis<T: NumericScalar>(
    plan: &WaveletPlan<T>,
    input: &[T],
    output: &mut [T],
    columns: usize,
    work: &mut [T],
    context: ExecutionContext<'_>,
) -> Result<bool, SciOxideError> {
    if plan.options.boundary != SignalBoundary::Periodic
        || !matches!(plan.filters.analysis_low.len(), 2 | 4)
        || plan
            .levels
            .iter()
            .any(|l| !l.input_len.is_multiple_of(2) || l.first != 0)
    {
        return Ok(false);
    }
    let width = plan.width * columns;
    let (a, rest) = work.split_at_mut(width);
    let b = &mut rest[..width];
    a[..plan.n * columns].copy_from_slice(&input[..plan.n * columns]);
    let low = &plan.filters.analysis_low;
    let high = &plan.filters.analysis_high;
    for level in &plan.levels {
        context.check()?;
        let (approximation, detail) =
            b[..2 * level.band_len * columns].split_at_mut(level.band_len * columns);
        for (j, (l, h)) in approximation
            .chunks_exact_mut(columns)
            .zip(detail.chunks_exact_mut(columns))
            .enumerate()
        {
            let row = |k: usize| {
                ((2 * j as i128 + k as i128 - plan.offset()).rem_euclid(level.input_len as i128)
                    as usize)
                    * columns
            };
            for start in (0..columns).step_by(256) {
                context.check()?;
                let end = (start + 256).min(columns);
                let l = &mut l[start..end];
                let h = &mut h[start..end];
                let x = |k| &a[row(k) + start..row(k) + end];
                match low.len() {
                    2 => {
                        let (l0, l1, h0, h1) = (
                            T::from_real(low[0]),
                            T::from_real(low[1]),
                            T::from_real(high[0]),
                            T::from_real(high[1]),
                        );
                        for (((l, h), &a), &b) in l.iter_mut().zip(h).zip(x(0)).zip(x(1)) {
                            *l = a * l0 + b * l1;
                            *h = a * h0 + b * h1;
                        }
                    }
                    4 => {
                        let l0 = T::from_real(low[0]);
                        let l1 = T::from_real(low[1]);
                        let l2 = T::from_real(low[2]);
                        let l3 = T::from_real(low[3]);
                        let h0 = T::from_real(high[0]);
                        let h1 = T::from_real(high[1]);
                        let h2 = T::from_real(high[2]);
                        let h3 = T::from_real(high[3]);
                        for (((((l, h), &a), &b), &c), &d) in
                            l.iter_mut().zip(h).zip(x(0)).zip(x(1)).zip(x(2)).zip(x(3))
                        {
                            *l = (a * l0 + b * l1) + (c * l2 + d * l3);
                            *h = (a * h0 + b * h1) + (c * h2 + d * h3);
                        }
                    }
                    _ => return Ok(false),
                }
            }
        }
        if !b[..2 * level.band_len * columns]
            .iter()
            .all(|v| v.has_finite_components())
        {
            return Ok(false);
        }
        let count = level.band_len * columns;
        let offset = level.detail_offset * columns;
        output[offset..offset + count].copy_from_slice(&b[count..2 * count]);
        a[..count].copy_from_slice(&b[..count]);
    }
    let count = plan.approximation * columns;
    output[..count].copy_from_slice(&a[..count]);
    Ok(true)
}
