//! Numerical execution with a prepared workspace, and borrowed adapters.
use super::*;
use crate::validation::length;
use scioxide_core::{
    AdjointOperator, AlgebraicScalar, BlockOperator, LinearOperator, NdViewMut, NumericScalar,
    SolveContext,
};

use crate::validation::sufficient;
fn conjugate<S: RealScalar>(x: Complex<S>, yes: bool) -> Complex<S> {
    Complex::new(x.re, if yes { -x.im } else { x.im })
}
// Plain sums would grow rounding with the number of terms: points per grid
// cell when spreading, width^rank products per target when interpolating.
// Both accumulators below keep the first-order error at eps = S::EPSILON times
// the summed magnitudes, the constant term of the documented rounding estimate.
//
// Grid cells receive one term at a time through memory, so Kahan's four
// additions per term are cheapest; corrections are folded in before the FFT.
fn kahan<S: RealScalar>(sum: &mut Complex<S>, correction: &mut Complex<S>, x: Complex<S>) {
    let y = x - *correction;
    let next = *sum + y;
    *correction = (next - *sum) - y;
    *sum = next;
}
fn finish<S: RealScalar>(grid: &mut [Complex<S>], correction: &[Complex<S>]) {
    for (sum, &correction) in grid.iter_mut().zip(correction) {
        *sum -= correction;
    }
}
// Interpolation adds each pair of weighted products plainly and compensates
// the pair sums of one target with branch-free TwoSum: the
// result s errs by (eps/2)(sum|a+b| + |s|) <= eps*sum(|a|+|b|), with half as
// many compensated steps as products and no serial Kahan chain per product.
fn compensated<S: RealScalar>(sum: &mut Complex<S>, correction: &mut Complex<S>, x: Complex<S>) {
    crate::numeric::compensated_term(&mut sum.re, &mut correction.re, x.re);
    crate::numeric::compensated_term(&mut sum.im, &mut correction.im, x.im);
}
#[expect(clippy::too_many_arguments)]
fn spread<S: RealScalar>(
    stencil: &Stencil<S>,
    input: impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
    grid: &mut [Complex<S>],
    correction: &mut [Complex<S>],
    kernel: &Kernel,
    weights: &mut [Complex<S>],
    tile: &mut [Complex<S>],
    adjoint: bool,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    grid.fill(Complex::<S>::ZERO);
    correction.fill(Complex::<S>::ZERO);
    if !stencil.bins.is_empty() {
        let side = kernel.width + SPREAD_BIN - 1;
        let [a, b, c] = stencil.axes[..] else {
            return Err(numerical(
                scioxide_core::NumericalIssue::Breakdown,
                "invalid spreading tile geometry",
            ));
        };
        for bin in &stencil.bins {
            for chunk in stencil.order[bin.start..bin.end].chunks(spread_chunk::<S>()) {
                context.check()?;
                tile.fill(Complex::<S>::ZERO);
                for &j in chunk {
                    stencil.weights(j, kernel, weights)?;
                    let value = input(j)? * conjugate(stencil.factor(j), adjoint);
                    let positions = &stencil.positions[j * 3..j * 3 + 3];
                    let x = positions[0].start - bin.origin[0];
                    let y = positions[1].start - bin.origin[1];
                    let z = positions[2].start - bin.origin[2];
                    crate::kernel::nufft_tile(
                        kernel.dispatch,
                        tile,
                        weights,
                        value,
                        side,
                        [(x, a.support), (y, b.support), (z, c.support)],
                    );
                }
                let mut x = bin.origin[0];
                for plane in tile.chunks_exact(side * side) {
                    let mut y = bin.origin[1];
                    for row in plane.chunks_exact(side) {
                        let mut z = bin.origin[2];
                        let mut left = row;
                        while !left.is_empty() {
                            let n = left.len().min(c.extent - z);
                            let index = x * a.stride + y * b.stride + z;
                            for ((sum, correction), &value) in grid[index..index + n]
                                .iter_mut()
                                .zip(&mut correction[index..index + n])
                                .zip(&left[..n])
                            {
                                kahan(sum, correction, value);
                            }
                            left = &left[n..];
                            z = 0;
                        }
                        y += 1;
                        if y == b.extent {
                            y = 0;
                        }
                    }
                    x += 1;
                    if x == a.extent {
                        x = 0;
                    }
                }
            }
        }
        finish(grid, correction);
        return Ok(());
    }
    // The adjoint of "multiply by factor, then spread" is "interpolate, then
    // multiply by conj(factor)" (and vice versa), so both helpers take the
    // `adjoint` flag only to conjugate the per-point factor. Cancellation is
    // polled every 256 points to bound latency without per-point overhead.
    for (ordinal, &j) in stencil.order.iter().enumerate() {
        let factor = stencil.factor(j);
        stencil.weights(j, kernel, weights)?;
        if ordinal.is_multiple_of(256) {
            context.check()?;
        }
        let value = input(j)? * conjugate(factor, adjoint);
        stencil.for_each_row(j, kernel.width, weights, |index, weight, row| {
            let value = value * weight;
            crate::kernel::nufft_spread(
                kernel.dispatch,
                &mut grid[index..index + row.len()],
                &mut correction[index..index + row.len()],
                row,
                value,
            );
        });
    }
    finish(grid, correction);
    Ok(())
}
fn interpolate<S: RealScalar>(
    stencil: &Stencil<S>,
    grid: &[Complex<S>],
    output: &mut [Complex<S>],
    kernel: &Kernel,
    weights: &mut [Complex<S>],
    adjoint: bool,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    for (ordinal, &j) in stencil.order.iter().enumerate() {
        let out = &mut output[j];
        let factor = stencil.factor(j);
        stencil.weights(j, kernel, weights)?;
        if ordinal.is_multiple_of(256) {
            context.check()?;
        }
        if let [a, b, c] = stencil.axes.as_slice() {
            let positions = &stencil.positions[j * 3..j * 3 + 3];
            let axes = [a, b, c].map(|axis| *axis);
            let geometry =
                core::array::from_fn(|d| (positions[d].start, axes[d].extent, axes[d].support));
            if let Some(value) = crate::kernel::nufft_interpolate3(
                kernel.dispatch,
                grid,
                weights,
                geometry,
                kernel.width,
            ) {
                *out = value * conjugate(factor, adjoint);
                continue;
            }
        }
        let (mut sum, mut correction) = (Complex::<S>::ZERO, Complex::<S>::ZERO);
        // An odd row's last product pairs with the next row's.
        let mut pending = None;
        stencil.for_each_row(j, kernel.width, weights, |index, weight, row| {
            let (pairs, last) = grid[index..index + row.len()].as_chunks::<2>();
            let (weight_pairs, last_weight) = row.as_chunks::<2>();
            for ([a, b], [wa, wb]) in pairs.iter().zip(weight_pairs) {
                let pair = *a * (weight * wa.re) + *b * (weight * wb.re);
                compensated(&mut sum, &mut correction, pair);
            }
            if let ([a], [w]) = (last, last_weight) {
                let product = *a * (weight * w.re);
                match pending.take() {
                    Some(previous) => compensated(&mut sum, &mut correction, previous + product),
                    None => pending = Some(product),
                }
            }
        });
        if let Some(product) = pending {
            compensated(&mut sum, &mut correction, product);
        }
        *out = (sum + correction) * conjugate(factor, adjoint);
    }
    Ok(())
}

impl<S: RealScalar> NufftPlan<S> {
    fn transform_grid(
        &self,
        grid: &mut [Complex<S>],
        work: &mut [Complex<S>],
        adjoint: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        // Separable N-D FFT over the oversampled grid, one axis at a time via
        // a contiguous lane copy. The adjoint of an unscaled DFT is the
        // unscaled DFT with the opposite sign, hence the XOR.
        let (lane, child_work) = work.split_at_mut(self.lane_len);
        let inverse = (self.spec.direction == FftDirection::Inverse) != adjoint;
        for ((&n, &stride), fft) in self
            .grid_shape
            .iter()
            .zip(&self.grid_strides)
            .zip(&self.fft)
        {
            for block in grid.chunks_exact_mut(n * stride) {
                for start in 0..stride {
                    if stride == 1 {
                        self.transform_lane(fft, block, child_work, inverse, context)?;
                    } else {
                        let lane = &mut lane[..n];
                        for (j, value) in lane.iter_mut().enumerate() {
                            *value = block[start + j * stride];
                        }
                        self.transform_lane(fft, lane, child_work, inverse, context)?;
                        for (j, &value) in lane.iter().enumerate() {
                            block[start + j * stride] = value;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn transform_lane(
        &self,
        fft: &ComplexCore<S>,
        lane: &mut [Complex<S>],
        work: &mut [Complex<S>],
        inverse: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.bounded {
            fft.transform_bounded(lane, work, inverse, S::ONE, context)
        } else if inverse {
            fft.transform(lane, work, true, S::ONE, context)
        } else {
            fft.transform(lane, work, false, S::ONE, context)
        }
    }
    fn embed(
        &self,
        input: impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        grid: &mut [Complex<S>],
    ) -> Result<(), SciOxideError> {
        grid.fill(Complex::<S>::ZERO);
        let mut result = Ok(());
        modes_each(&self.modes, |j, index, weight| {
            if result.is_ok() {
                result = input(j).map(|value| grid[index] = value * (weight * self.scale));
            }
        });
        result
    }
    fn extract(&self, grid: &[Complex<S>], output: &mut [Complex<S>]) {
        modes_each(&self.modes, |j, index, weight| {
            output[j] = grid[index] * (weight * self.scale)
        });
    }
    // Shared execution kernel.
    fn evaluate(
        &self,
        input: impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        scratch: &mut [Complex<S>],
        adjoint: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        sufficient(scratch.len(), self.scratch_len)?;
        context.check()?;
        let (n, m) = if adjoint {
            (self.output_len, self.input_len)
        } else {
            (self.input_len, self.output_len)
        };
        let mut magnitude = S::ZERO;
        for j in 0..n {
            let value = input(j)?;
            crate::validation::check_complex_slice_finite(
                &[value],
                scioxide_core::NonFiniteOrigin::Input,
            )?;
            magnitude = magnitude.max(value.re.abs()).max(value.im.abs());
        }
        let (candidate, work) = scratch.split_at_mut(self.input_len.max(self.output_len));
        let candidate = &mut candidate[..m];
        if magnitude == S::ZERO || m == 0 {
            candidate.fill(Complex::<S>::ZERO);
            context.check()?;
            return Ok(());
        }
        // One binary input scale covers spreading, deconvolution and every
        // grid axis. Mode factors already contain the requested normalization.
        let exponent = crate::scaling::range_exponent(magnitude) + self.headroom;
        let normalized = |j| {
            let z = input(j)?;
            Ok(Complex::new(z.re.scalbn(-exponent), z.im.scalbn(-exponent)))
        };
        let (weights, work) = work.split_at_mut(self.dimension() * self.kernel.width);
        let (tile, work) = work.split_at_mut(self.tile_len);
        let (grid, work) = work.split_at_mut(self.diagnostics.grid_points);
        let (correction, work) = work.split_at_mut(self.diagnostics.grid_points);
        // Pipelines (D = diagonal deconvolution on modes, F = grid FFT):
        //   type 1:        spread sources -> F -> extract modes with D
        //   type 2:        embed modes with D -> F -> interpolate at points
        //   type 3:        spread sources onto the base grid -> embed with D
        //                  -> F -> interpolate at rescaled frequencies
        // Each adjoint runs the transposed pipeline in reverse order, which
        // is why type-1 adjoint shares the type-2 path and vice versa.
        match (&self.geometry, adjoint) {
            (PreparedGeometry::One(source), false) => {
                spread(
                    source,
                    normalized,
                    grid,
                    correction,
                    &self.kernel,
                    weights,
                    tile,
                    false,
                    context,
                )?;
                self.transform_grid(grid, work, false, context)?;
                self.extract(grid, candidate);
            }
            (PreparedGeometry::Two(target), true) => {
                spread(
                    target,
                    normalized,
                    grid,
                    correction,
                    &self.kernel,
                    weights,
                    tile,
                    true,
                    context,
                )?;
                self.transform_grid(grid, work, true, context)?;
                self.extract(grid, candidate);
            }
            (PreparedGeometry::Two(stencil), false) | (PreparedGeometry::One(stencil), true) => {
                self.embed(normalized, grid)?;
                self.transform_grid(grid, work, adjoint, context)?;
                interpolate(
                    stencil,
                    grid,
                    candidate,
                    &self.kernel,
                    weights,
                    adjoint,
                    context,
                )?;
            }
            (PreparedGeometry::Three { source, target }, _) => {
                let (base, work) = work.split_at_mut(self.base_len);
                let (base_correction, work) = work.split_at_mut(self.base_len);
                if adjoint {
                    spread(
                        target,
                        normalized,
                        grid,
                        correction,
                        &self.kernel,
                        weights,
                        tile,
                        true,
                        context,
                    )?;
                    self.transform_grid(grid, work, true, context)?;
                    self.extract(grid, base);
                    interpolate(
                        source,
                        base,
                        candidate,
                        &self.kernel,
                        weights,
                        true,
                        context,
                    )?;
                } else {
                    spread(
                        source,
                        normalized,
                        base,
                        base_correction,
                        &self.kernel,
                        weights,
                        tile,
                        false,
                        context,
                    )?;
                    self.embed(|j| Ok(base[j]), grid)?;
                    self.transform_grid(grid, work, false, context)?;
                    interpolate(
                        target,
                        grid,
                        candidate,
                        &self.kernel,
                        weights,
                        false,
                        context,
                    )?;
                }
            }
        }
        crate::scaling::restore_complex(candidate, exponent, S::INFINITY)?;
        context.check()
    }

    // Borrowed operator calls use the same child admission as owned execution.
    fn apply_borrowed(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut [Complex<S>],
        adjoint: bool,
        scope: scioxide_core::SolveScope<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let (n, m) = if adjoint {
            (self.output_len, self.input_len)
        } else {
            (self.input_len, self.output_len)
        };
        length(input.len(), n)?;
        length(output.len(), m)?;
        sufficient(workspace.len(), self.scratch_len)?;
        scope.execution().check()?;
        scope.run(
            &mut scioxide_core::SolveState::new(),
            crate::workspace::identity(self.input_len),
            scioxide_core::SolveBudgets::unbounded(),
            |child| {
                crate::resource::admit_operator(child)?;
                self.evaluate(|j| Ok(input[j]), workspace, adjoint, child.execution())?;
                output.copy_from_slice(&workspace[..m]);
                Ok(())
            },
        )
    }

    // Owned output for the allocating conveniences; prepared plans publish
    // only caller-storage forms.
    pub(crate) fn execute_owned(
        &self,
        input: crate::FftInput<'_, S>,
    ) -> Result<Vec<Complex<S>>, SciOxideError>
    where
        Complex<S>: NumericScalar,
    {
        let mut output = try_filled(
            self.output_len,
            Complex::<S>::ZERO,
            scioxide_core::quantity!("nufft_output"),
        )?;
        self.execute_into(
            input,
            &mut output,
            &mut self.workspace(())?,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Applies the transform to one real or complex vector and writes the result
    /// to `output`.
    ///
    /// The input may be strided or conjugated. The call counts one
    /// linear-operator evaluation in `scope`. For a returned `Vec`, use
    /// [`nufft_type1`](crate::nufft_type1), [`nufft_type2`](crate::nufft_type2)
    /// or [`nufft_type3`](crate::nufft_type3); for the adjoint, use
    /// [`AdjointOperator`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the input or output length differs from
    ///   [`Self::input_len`] or [`Self::output_len`], or `workspace` is too
    ///   small.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `output` is not complex.
    /// - `NonFinite` if the input contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline, and budget errors from
    ///   `scope`.
    pub fn execute_into<'i, 'o, 'scope, 'cap>(
        &self,
        input: impl Into<crate::FftInput<'i, S>>,
        output: impl Into<crate::FftOutput<'o, S>>,
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError>
    where
        Complex<S>: NumericScalar,
    {
        let input = crate::io::FullInput::try_from(input.into())?;
        let input_len = input.len();
        let crate::FftOutput::Complex(mut output) = output.into() else {
            return Err(invalid(
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                scioxide_core::argument!("output"),
                "NUFFT output must be complex",
            ));
        };
        length(input_len, self.input_len)?;
        length(output.len(), self.output_len)?;
        workspace.run(
            &self.workspace_requirements(())?,
            self.input_len,
            scioxide_core::SolveBudgets::unbounded(),
            scope.into(),
            |scratch, child| {
                crate::resource::admit_operator(child)?;
                let context = child.execution();
                match input {
                    crate::io::FullInput::Complex(view) => {
                        self.evaluate(|j| view.get(j), scratch, false, context)?
                    }
                    crate::io::FullInput::Real(view) => self.evaluate(
                        |j| Ok(Complex::new(view.get(j)?, S::ZERO)),
                        scratch,
                        false,
                        context,
                    )?,
                }
                for (j, &v) in scratch[..self.output_len].iter().enumerate() {
                    output.set(j, v)?;
                }
                Ok(())
            },
        )
    }
    /// Transforms every lane of `input` along `axis` into `output`; the other
    /// axes are batch axes.
    ///
    /// The extents along `axis` must equal [`Self::input_len`] and
    /// [`Self::output_len`]. Either every lane is written or, on error, none
    /// is. A call with at least one lane counts one linear-operator evaluation.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis` is not an axis of `input`.
    /// - `DimensionMismatch` if the extents or batch shapes do not match, or
    ///   `workspace` is too small for this request.
    /// - `NonFinite` if the input contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline, and budget errors from
    ///   `scope`.
    pub fn execute_axis_into<'scope, 'cap, T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        output: NdViewMut<'_, '_, Complex<S>>,
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute(self, input, axis, output, workspace, scope.into())
    }
}

impl<S: RealScalar> crate::axis::AxisPlan<S> for NufftPlan<S> {
    fn lane_lens(&self) -> (usize, usize) {
        (self.input_len, self.output_len)
    }
    fn scratch(&self) -> Result<FftWorkspaceRequirements, SciOxideError> {
        let scratch = FftWorkspaceRequirements::new::<S>(self.scratch_len)?;
        let bytes = |len: usize| len * size_of::<Complex<S>>();
        scratch.with_limit(crate::workspace::nested_byte_limit(
            self.spec.limits.workspace_bytes,
            scratch.bytes(),
            self.fft.iter().map(|child| {
                if self.bounded {
                    (
                        child.bounded_workspace_byte_limit(),
                        bytes(child.bounded_scratch_len()),
                    )
                } else {
                    (child.workspace_byte_limit(), bytes(child.scratch_len()))
                }
            }),
        ))
    }
    fn workspace_bytes(&self) -> usize {
        self.spec.limits.workspace_bytes
    }
    // One operator evaluation covers every lane, as for `execute_into`.
    fn admit(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        crate::resource::admit_operator(context)
    }
    fn lane<'w>(
        &self,
        source: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(|j| Ok(source[j]), scratch, false, context.execution())?;
        Ok(&scratch[..self.output_len])
    }
}

impl<S: RealScalar> LinearOperator<Complex<S>> for NufftPlan<S>
where
    Complex<S>: NumericScalar,
{
    fn rows(&self) -> usize {
        self.output_len
    }
    fn cols(&self) -> usize {
        self.input_len
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch_len)
    }
    fn apply_into(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut [Complex<S>],
    ) -> Result<(), SciOxideError> {
        self.apply_borrowed(
            input,
            output,
            workspace,
            false,
            scioxide_core::ExecutionContext::sequential_native().into(),
        )
    }
    fn apply_into_scoped(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut [Complex<S>],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, scioxide_core::Complex<S>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_borrowed(input, output, workspace, false, (&mut **context).into())
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: RealScalar> AdjointOperator<Complex<S>> for NufftPlan<S>
where
    Complex<S>: NumericScalar,
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch_len)
    }
    fn apply_adjoint_into(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut [Complex<S>],
    ) -> Result<(), SciOxideError> {
        self.apply_borrowed(
            input,
            output,
            workspace,
            true,
            scioxide_core::ExecutionContext::sequential_native().into(),
        )
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut [Complex<S>],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, scioxide_core::Complex<S>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_borrowed(input, output, workspace, true, (&mut **context).into())
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: RealScalar> BlockOperator<Complex<S>> for NufftPlan<S>
where
    Complex<S>: NumericScalar,
{
    fn apply_block_workspace_len(&self, batches: usize) -> Result<usize, SciOxideError> {
        if batches == 0 {
            return Ok(0);
        }
        let len = checked_add(
            checked_mul(
                self.output_len,
                batches,
                scioxide_core::quantity!("nufft_block_outputs"),
            )?,
            self.scratch_len,
            scioxide_core::quantity!("nufft_block_workspace"),
        )?;
        admit_bytes(
            checked_mul(
                len,
                size_of::<Complex<S>>(),
                scioxide_core::quantity!("nufft_block_workspace_bytes"),
            )?,
            self.spec.limits.workspace_bytes,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        Ok(len)
    }
}
