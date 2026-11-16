//! Native smooth transforms over consecutive independent lanes.
//!
//! Complex runs batch the existing one-lane leaves over consecutive buffers.
//! Real/trigonometric adapters on Scalar and AVX2 instead use block rows with
//! exactly the same twiddle arithmetic, avoiding input/output transposes.
//! Other backends keep the original leaves through consecutive lane buffers.
use scioxide_core::{Complex, RealScalar};

use super::block::{Block, BlockLanes, BlockPass, Run, WIDTH};
use super::{StagePlan, admit};

impl<S: RealScalar> StagePlan<S> {
    /// Backends whose exact one-lane products have an across-lane adapter.
    pub(crate) fn native_rows_supported(&self) -> bool {
        matches!(
            self.dispatch.backend(),
            scioxide_core::backend::KernelBackend::Scalar
                | scioxide_core::backend::KernelBackend::Avx2Fma
        )
    }

    /// Transform digit-ordered block rows with the selected backend's exact
    /// products, including fused twiddles, scalar tails and unit products.
    pub(crate) fn transform_native_rows(&self, rows: &mut [Block<S, WIDTH>], inverse: bool) {
        admit(rows.len() >= self.n && self.native_rows_supported());
        super::run_pass(
            self.dispatch,
            NativeRows {
                plan: self,
                rows,
                inverse,
            },
        );
    }
    /// Inverse gather map for adapters that stage directly in digit order.
    pub(crate) fn digit_rows(
        &self,
    ) -> Result<alloc::vec::Vec<usize>, scioxide_core::SciOxideError> {
        let mut order = scioxide_core::backend::try_filled(
            self.n,
            0,
            scioxide_core::quantity!("fft_digit_rows"),
        )?;
        for (to, &from) in self.permutation.iter().enumerate() {
            order[from] = to;
        }
        Ok(order)
    }

    /// Finish complete lanes already checked and staged in digit order.
    /// Keeping the final leaf preserves its unit-twiddle products and zeros.
    pub(crate) fn transform_staged_batch(
        &self,
        data: &mut [Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        admit(data.len().is_multiple_of(self.n) && output.len() >= data.len() && scale.is_finite());
        let output = &mut output[..data.len()];
        if self.n == 1 {
            for (out, &value) in output.iter_mut().zip(data.iter()) {
                *out = crate::scaling::scale_singleton(crate::validation::bounded(value), scale);
            }
            return;
        }
        let first = 0;
        // NEON's ordinary gather uses the scalar width-one butterfly. Its
        // in-place vector stage would multiply unit twiddles instead, changing
        // signed zeros. Retain the gather arithmetic when staging bypasses it.
        #[cfg(target_arch = "aarch64")]
        let first = if self.dispatch.backend() == scioxide_core::backend::KernelBackend::Neon
            && self.stage_count() > 1
        {
            let stage = &self.stages[0];
            if inverse {
                super::scalar::execute::<S, true>(
                    data,
                    stage.radix,
                    stage.width,
                    &stage.twiddles,
                    &stage.roots,
                );
            } else {
                super::scalar::execute::<S, false>(
                    data,
                    stage.radix,
                    stage.width,
                    &stage.twiddles,
                    &stage.roots,
                );
            }
            1
        } else {
            first
        };
        self.run(data, first..self.stage_count() - 1, inverse);
        for (from, to) in data
            .chunks_exact(self.n)
            .zip(output.chunks_exact_mut(self.n))
        {
            self.finish(from, to, inverse, scale);
        }
    }
    /// Native bounded lanes, with direct access to adjacent contiguous rows.
    pub(crate) fn transform_native_lanes(
        &self,
        values: &mut [Complex<S>],
        run: Run,
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        admit(run.len == self.n && scratch.len() >= 2 * WIDTH * self.n);
        super::run_pass(
            self.dispatch,
            NativeLanes {
                plan: self,
                values,
                run,
                scratch,
                inverse,
                scale,
            },
        );
    }
    /// Transform consecutive natural-order lanes using equally sized scratch.
    /// Each lane has exactly the arithmetic of a bounded one-lane execution.
    /// All metadata is prepared; this entry allocates and polls nothing.
    pub(crate) fn transform_batch(
        &self,
        values: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        admit(
            values.len().is_multiple_of(self.n)
                && scratch.len() >= values.len()
                && scale.is_finite(),
        );
        if self.n == 1 {
            for value in values {
                *value = crate::scaling::scale_singleton(crate::validation::bounded(*value), scale);
            }
            return;
        }
        let scratch = &mut scratch[..values.len()];
        let mut completed = 0;
        for (input, output) in values
            .chunks_exact(self.n)
            .zip(scratch.chunks_exact_mut(self.n))
        {
            completed = self.gather_bounded(input, output, inverse, false);
        }
        self.run(scratch, completed..self.stage_count() - 1, inverse);
        for (input, output) in scratch
            .chunks_exact(self.n)
            .zip(values.chunks_exact_mut(self.n))
        {
            self.finish(input, output, inverse, scale);
        }
    }
}

struct NativeRows<'p, 'r, S: RealScalar> {
    plan: &'p StagePlan<S>,
    rows: &'r mut [Block<S, WIDTH>],
    inverse: bool,
}
impl<S: RealScalar> BlockPass for NativeRows<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        super::tile::transform_rows(self.plan, self.rows, self.inverse, true);
    }
}

struct NativeLanes<'p, 'v, S: RealScalar> {
    plan: &'p StagePlan<S>,
    values: &'v mut [Complex<S>],
    run: Run,
    scratch: &'v mut [Complex<S>],
    inverse: bool,
    scale: S,
}

impl<S: RealScalar> BlockPass for NativeLanes<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        let n = self.plan.n;
        let (data, work) = self.scratch.split_at_mut(WIDTH * n);
        for block in (0..self.run.count).step_by(WIDTH) {
            let count = (self.run.count - block).min(WIDTH);
            if self.run.stride == 1 && self.run.step == n {
                let first = self.run.first + block * n;
                self.plan.transform_batch(
                    &mut self.values[first..first + count * n],
                    work,
                    self.inverse,
                    self.scale,
                );
                continue;
            }
            let lanes = BlockLanes::new(self.run, block);
            let data = &mut data[..count * n];
            for (i, &source) in self.plan.permutation.iter().enumerate() {
                let input = lanes.read(self.values, source);
                for (row, value) in data.chunks_exact_mut(n).zip(input) {
                    row[i] = crate::validation::bounded(value);
                }
            }
            self.plan
                .transform_staged_batch(data, work, self.inverse, self.scale);
            for i in 0..n {
                lanes.write(
                    self.values,
                    i,
                    core::array::from_fn(|lane| work[lane.min(count - 1) * n + i]),
                );
            }
        }
    }
}
