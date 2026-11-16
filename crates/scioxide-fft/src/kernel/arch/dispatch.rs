//! Unsafe FFT dispatch behind the immutable stage-plan proofs.
//!
//! Every safe entry checks borrowed geometry before invoking preselected
//! unsafe leaves. Stage metadata and permutation indices stay private.

use super::super::*;

/// Run `$body` with `$lane` bound to the component type of scalar `$scalar`.
macro_rules! by_lane {
    ($scalar:ty, $lane:ident => $body:expr) => {
        match <$scalar>::complex_slice(&[]) {
            ComplexSlice::F32(_) => {
                type $lane = f32;
                $body
            }
            ComplexSlice::F64(_) => {
                type $lane = f64;
                $body
            }
        }
    };
}

impl<S: RealScalar> Stage<S> {
    /// # Safety
    /// `data` must contain whole blocks of this stage, with its selected lane.
    unsafe fn apply<T: Lane>(&self, data: &mut [Complex<T>], inverse: bool) {
        // SAFETY: the caller proves whole stage blocks; Stage::new proves the selected
        // ISA, supported radix, specialization and exact twiddle/root lengths.
        unsafe {
            let kernels = T::kernels(&self.leaves);
            let leaf = if inverse {
                kernels.inverse
            } else {
                kernels.forward
            };
            (leaf.0)(
                data,
                self.radix,
                self.width,
                T::slice(&self.twiddles),
                T::slice(&self.roots),
            );
        }
    }
    /// # Safety
    /// The validated plan must pair these consecutive radix-4 stages and
    /// `data` must contain whole blocks of the second stage.
    unsafe fn apply_pair<T: Lane>(&self, second: &Self, data: &mut [Complex<T>], inverse: bool) {
        // SAFETY: the caller proves whole second-stage blocks and the validated width
        // chain; StagePlan::new marks only consecutive radix-4 stages as paired.
        unsafe {
            let kernels = T::kernels(&self.leaves);
            let (first_twiddles, second_twiddles) =
                (T::slice(&self.twiddles), T::slice(&second.twiddles));
            let fused = if inverse {
                kernels.fused_stage_inverse
            } else {
                kernels.fused_stage_forward
            };
            if let Some(fused) = fused {
                (fused.0)(data, first_twiddles, second_twiddles);
            } else {
                // Each second-stage block is four whole first-stage blocks.
                for block in data.chunks_exact_mut(second.block()) {
                    self.apply(block, inverse);
                    second.apply(block, inverse);
                }
            }
        }
    }
}

impl<S: RealScalar> StagePlan<S> {
    /// Gather ordinary input through the preselected fused leaves.
    pub(crate) fn gather(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        include_last: bool,
    ) -> usize {
        self.gather_impl(input, output, inverse, include_last, false)
    }
    /// Check bounded input in the same gathered loads as the first butterflies.
    pub(crate) fn gather_bounded(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        include_last: bool,
    ) -> usize {
        if self.stages.is_empty() {
            admit(input.len() == self.n && output.len() == self.n);
            crate::validation::bounded(input[0]);
        }
        self.gather_impl(input, output, inverse, include_last, true)
    }
    /// Gather full buffers, optionally fusing the leading stages.
    fn gather_impl(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        include_last: bool,
        bounded: bool,
    ) -> usize {
        admit(input.len() == self.n && output.len() == self.n);
        let prefix = if include_last {
            self.gather_all
        } else {
            self.gather_prefix
        };
        // SAFETY: admission proves full buffers; immutable construction proves
        // the gather geometry, permutation bounds, tables and selected ISA.
        unsafe {
            by_lane!(S, T => self.gather_typed::<T>(T::slice(input), T::slice_mut(output), inverse, prefix, bounded))
        }
    }
    /// # Safety
    /// Both buffers must have the plan length and selected scalar lane;
    /// `prefix` must be one of the plan's prepared gather choices.
    unsafe fn gather_typed<T: Lane>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        inverse: bool,
        prefix: GatherPrefix,
        bounded: bool,
    ) -> usize {
        // SAFETY: gather admitted both full buffers; StagePlan::new and Permutation::new
        // prove the complete gather groups, in-range sources, tables and selected ISA.
        unsafe {
            let Some(first) = self.stages.first() else {
                output.copy_from_slice(input);
                return 0;
            };
            let kernels = T::kernels(&first.leaves);
            match prefix {
                GatherPrefix::Permutation => {
                    let leaf = if bounded {
                        kernels.permute.1
                    } else {
                        kernels.permute.0
                    };
                    leaf(input, output, &self.permutation);
                    0
                }
                GatherPrefix::Single => {
                    let leaf = if inverse {
                        kernels.gather_inverse
                    } else {
                        kernels.gather_forward
                    };
                    let leaf = if bounded { leaf.1 } else { leaf.0 };
                    leaf(
                        input,
                        output,
                        &self.permutation,
                        first.radix,
                        T::slice(&first.roots),
                    );
                    1
                }
                GatherPrefix::Pair | GatherPrefix::Triple => {
                    let leaf = if inverse {
                        kernels.fused_gather_inverse
                    } else {
                        kernels.fused_gather_forward
                    };
                    let Some(leaf) = leaf else {
                        unreachable!("fused gather selection requires a fused leaf")
                    };
                    let leaf = if bounded { leaf.1 } else { leaf.0 };
                    leaf(
                        input,
                        output,
                        &self.permutation,
                        T::slice(&self.stages[1].twiddles),
                    );
                    if matches!(prefix, GatherPrefix::Triple) {
                        self.stages[2].apply(output, inverse);
                        3
                    } else {
                        2
                    }
                }
            }
        }
    }
    /// Run a validated range over whole blocks of its final stage.
    pub(crate) fn run(&self, data: &mut [Complex<S>], stages: Range<usize>, inverse: bool) {
        admit(stages.start <= stages.end && stages.end <= self.stages.len());
        if stages.is_empty() {
            return;
        }
        admit(
            data.len() == self.n
                || data
                    .len()
                    .is_multiple_of(self.stages[stages.end - 1].block()),
        );
        by_lane!(S, T => {
            let data = T::slice_mut(data);
            let mut remaining = &self.stages[stages];
            while let Some((first, tail)) = remaining.split_first() {
                if let Some((second, rest)) = tail.split_first()
                    && first.pair_with_next
                {
                    // SAFETY: admission proves whole last-stage blocks; the validated
                    // width chain makes them whole blocks of this paired stage.
                    unsafe { first.apply_pair(second, data, inverse) };
                    remaining = rest;
                } else {
                    // SAFETY: admission and the validated width chain prove whole
                    // blocks; Stage::new proves matching tables and selected ISA.
                    unsafe { first.apply(data, inverse) };
                    remaining = tail;
                }
            }
        });
    }
    /// Apply a required final stage into a full scaled output.
    pub(crate) fn finish(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        admit(
            input.len() == self.n
                && output.len() == self.n
                && scale.is_finite()
                && !self.stages.is_empty(),
        );
        let Some(last) = self.stages.last() else {
            unreachable!("finish admission requires a final stage");
        };
        by_lane!(S, T => {
            let kernels = T::kernels(&last.leaves);
            let leaf = if inverse {
                kernels.final_inverse
            } else {
                kernels.final_forward
            };
            // SAFETY: admission proves full equal-length buffers and finite scale;
            // StagePlan::new proves final geometry, tables and selected ISA.
            unsafe { (leaf.0)(
                T::slice(input),
                T::slice_mut(output),
                last.radix,
                last.width,
                T::slice(&last.twiddles),
                T::slice(&last.roots),
                T::value(scale),
            ) };
        });
    }
}
