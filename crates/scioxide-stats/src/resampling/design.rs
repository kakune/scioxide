//! Plan validation, counting and unranking of permutations, and random
//! resampling indices.
use super::runner::Slot;
use super::*;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    PermutationEnumerator, checked_binomial, checked_factorial, permutation_unrank_into,
};
use scioxide_random::{shuffle, uniform_index};

impl<S: RealScalar> ResamplingPlan<S> {
    /// Creates a plan for groups of sizes `lengths`, after checking them
    /// against `method` and `options`.
    ///
    /// The statistic receives indices into the concatenation of the groups
    /// ([`ResampledIndices`]). For an exact permutation test, the plan counts
    /// the assignments. For the jackknife, the number of replicates is the
    /// number of deletions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `lengths` is empty or has a zero, or the number of
    ///   replicates or the batch size is zero.
    /// - `InvalidInput` if a `Paired` design or a paired permutation kind has
    ///   groups of unequal length.
    /// - `InvalidInput` if a block design has more than one group or a block
    ///   length outside `1..=n`.
    /// - `InvalidInput` if BCa or the jackknife uses a block design or has a
    ///   group with fewer than two observations.
    /// - `InvalidInput` if a bootstrap has fewer than two replicates.
    /// - `InvalidInput` if a [`PermutationKind::Independent`] or
    ///   [`PermutationKind::PairedLabels`] test has fewer than two groups.
    /// - `InvalidInput` if a Monte Carlo test has more than one group.
    /// - `ConvergenceFailure` if the exact assignments or jackknife deletions
    ///   outnumber [`ResamplingOptions::replicates`].
    /// - `SizeOverflow` if the number of exact assignments overflows `usize`.
    pub fn new(
        lengths: &[usize],
        method: ResamplingMethod,
        options: ResamplingOptions<S>,
    ) -> Result<Self, SciOxideError> {
        if lengths.is_empty()
            || lengths.contains(&0)
            || options.replicates == 0
            || options.batch_size == 0
        {
            return Err(invalid(
                scioxide_core::argument!("lengths"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nonempty groups and positive replicate/batch counts required",
            ));
        }
        for &length in lengths {
            crate::error_helpers::count::<S>(length)?;
        }
        let mut offsets = try_filled(
            checked_add(
                lengths.len(),
                1,
                scioxide_core::quantity!("resampling_groups"),
            )?,
            0,
            scioxide_core::quantity!("resampling_offsets"),
        )?;
        for (i, &n) in lengths.iter().enumerate() {
            offsets[i + 1] =
                checked_add(offsets[i], n, scioxide_core::quantity!("resampling_length"))?;
        }
        let equal = lengths.iter().all(|&n| n == lengths[0]);
        let mut deletions = 0;
        let mut replicates = options.replicates;
        match method {
            ResamplingMethod::Bootstrap { design, .. } | ResamplingMethod::Jackknife { design } => {
                validate_design(lengths, design, equal)?;
            }
            _ => {}
        }
        let design = match method {
            ResamplingMethod::Bootstrap {
                design,
                method: BootstrapMethod::Bca,
                ..
            }
            | ResamplingMethod::Jackknife { design } => Some(design),
            _ => None,
        };
        if let Some(design) = design {
            if matches!(
                design,
                BootstrapDesign::MovingBlock { .. } | BootstrapDesign::CircularBlock { .. }
            ) {
                return Err(invalid(
                    scioxide_core::argument!("method"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "block BCa and block jackknife are unsupported",
                ));
            }
            if lengths.iter().any(|&n| n < 2) {
                return Err(invalid(
                    scioxide_core::argument!("lengths"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "jackknife/BCa needs at least two units per group",
                ));
            }
            deletions = if design == BootstrapDesign::Paired {
                lengths[0]
            } else {
                offsets[lengths.len()]
            };
            if matches!(method, ResamplingMethod::Jackknife { .. }) {
                replicates = deletions;
                deletions = 0;
            }
        }
        match method {
            ResamplingMethod::Bootstrap { .. } if replicates < 2 => {
                return Err(invalid(
                    scioxide_core::argument!("method"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "bootstrap needs at least two replicates",
                ));
            }
            ResamplingMethod::Permutation { kind, mode, .. } => {
                if kind != PermutationKind::Independent && !equal {
                    return Err(invalid(
                        scioxide_core::argument!("lengths"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "paired permutations require equal group lengths",
                    ));
                }
                if kind != PermutationKind::Pairings && lengths.len() < 2 {
                    return Err(invalid(
                        scioxide_core::argument!("lengths"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "label permutations require at least two groups",
                    ));
                }
                if mode == PermutationMode::Exact {
                    replicates = match kind {
                        PermutationKind::Independent => multinomial(lengths)?,
                        PermutationKind::PairedLabels => {
                            power(checked_factorial(lengths.len())?, lengths[0])?
                        }
                        PermutationKind::Pairings => {
                            power(checked_factorial(lengths[0])?, lengths.len())?
                        }
                    };
                }
            }
            ResamplingMethod::MonteCarlo { .. } if lengths.len() != 1 => {
                return Err(invalid(
                    scioxide_core::argument!("lengths"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Monte Carlo samples use one flat group",
                ));
            }
            _ => {}
        }
        crate::error_helpers::work(
            replicates,
            options.replicates,
            scioxide_core::BudgetKind::Sample,
            "exact or jackknife replicates exceed options.replicates",
        )?;
        crate::error_helpers::count::<S>(replicates)?;
        let tolerance = match options.tie_tolerance {
            Some(t) => t,
            None => Tolerance::new(S::ZERO, S::EPSILON * S::try_from_f64_lossy(100.0)?)?,
        };
        Ok(Self {
            statistic_scratch: 0,
            offsets,
            method,
            options,
            replicates,
            deletions,
            tolerance,
        })
    }
    /// Returns the method.
    pub const fn method(&self) -> ResamplingMethod {
        self.method
    }
    /// Returns the number of replicate statistics, which is the length of the
    /// output slice. It is `B` for random replicates, the number of
    /// assignments for an exact permutation test and the number of deletions
    /// for the jackknife.
    pub const fn replicates(&self) -> usize {
        self.replicates
    }
    /// Returns the length of the scratch slice of each worker; see
    /// [`Self::with_statistic_scratch`]. The default is 0.
    pub const fn statistic_scratch_elements(&self) -> usize {
        self.statistic_scratch
    }
    /// Sets the length of the scratch slice passed to the statistic of
    /// [`ResamplingProblem::IndexedWorkspace`] and
    /// [`ResamplingProblem::MonteCarloWorkspace`].
    ///
    /// Each worker has its own slice, which is reused for every call,
    /// including the observed statistic. It counts toward
    /// [`ResamplingOptions::max_workspace_bytes`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed
    ///   [`ResamplingOptions::max_workspace_bytes`].
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    pub fn with_statistic_scratch(mut self, elements: usize) -> Result<Self, SciOxideError> {
        self.statistic_scratch = elements;
        crate::error_helpers::limit(
            self.workspace_bytes()?,
            self.options.max_workspace_bytes,
            scioxide_core::quantity!("design_bytes"),
        )?;
        Ok(self)
    }
    pub(super) fn total(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }
    pub(super) fn groups(&self) -> usize {
        self.offsets.len() - 1
    }
    pub(super) fn jobs(&self) -> usize {
        self.replicates.max(self.deletions)
    }
    pub(super) fn paired(&self) -> bool {
        matches!(
            self.method,
            ResamplingMethod::Bootstrap {
                design: BootstrapDesign::Paired,
                ..
            } | ResamplingMethod::Jackknife {
                design: BootstrapDesign::Paired
            }
        )
    }
    pub(super) fn identity(&self, slot: &mut Slot<S>) {
        PermutationEnumerator::identity_into(&mut slot.indices);
        slot.offsets.copy_from_slice(&self.offsets);
    }
    pub(super) fn delete(&self, deletion: usize, slot: &mut Slot<S>) {
        let mut used = 0;
        slot.offsets[0] = 0;
        for (g, pair) in self.offsets.windows(2).enumerate() {
            for index in pair[0]..pair[1] {
                let omit = if self.paired() {
                    index - pair[0] == deletion
                } else {
                    index == deletion
                };
                if !omit {
                    slot.indices[used] = index;
                    used += 1;
                }
            }
            slot.offsets[g + 1] = used;
        }
    }
    pub(super) fn sample(
        &self,
        replicate: usize,
        rng: &mut dyn BitGenerator,
        slot: &mut Slot<S>,
    ) -> Result<(), SciOxideError> {
        self.identity(slot);
        match self.method {
            ResamplingMethod::Bootstrap { design, .. } => {
                bootstrap_indices(&self.offsets, design, rng, &mut slot.indices)
            }
            ResamplingMethod::Jackknife { .. } => {
                self.delete(replicate, slot);
                Ok(())
            }
            ResamplingMethod::Permutation { kind, mode, .. } => {
                permutation(&self.offsets, kind, mode, replicate, rng, slot)
            }
            ResamplingMethod::MonteCarlo { .. } => Ok(()),
        }
    }
}
fn validate_design(
    lengths: &[usize],
    design: BootstrapDesign,
    equal: bool,
) -> Result<(), SciOxideError> {
    match design {
        BootstrapDesign::Paired if !equal => Err(invalid(
            scioxide_core::argument!("lengths"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "paired bootstrap requires equal group lengths",
        )),
        BootstrapDesign::MovingBlock { length } | BootstrapDesign::CircularBlock { length }
            if lengths.len() != 1 || length == 0 || length > lengths[0] =>
        {
            Err(invalid(
                scioxide_core::argument!("method"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "block length must lie in 1..=n for a single group",
            ))
        }
        _ => Ok(()),
    }
}
fn bootstrap_indices(
    offsets: &[usize],
    design: BootstrapDesign,
    mut rng: &mut dyn BitGenerator,
    indices: &mut [usize],
) -> Result<(), SciOxideError> {
    match design {
        BootstrapDesign::MovingBlock { length } | BootstrapDesign::CircularBlock { length } => {
            let n = indices.len();
            let circular = matches!(design, BootstrapDesign::CircularBlock { .. });
            for block in indices.chunks_mut(length) {
                let start = uniform_index(if circular { n } else { n - length + 1 }, &mut rng)?;
                for (i, target) in block.iter_mut().enumerate() {
                    *target = if i >= n - start {
                        i - (n - start)
                    } else {
                        start + i
                    };
                }
            }
        }
        BootstrapDesign::Paired => {
            let n = offsets[1];
            for row in 0..n {
                let index = uniform_index(n, &mut rng)?;
                for &start in &offsets[..offsets.len() - 1] {
                    indices[start + row] = start + index;
                }
            }
        }
        BootstrapDesign::Independent | BootstrapDesign::Stratified => {
            for pair in offsets.windows(2) {
                for index in &mut indices[pair[0]..pair[1]] {
                    *index = pair[0] + uniform_index(pair[1] - pair[0], &mut rng)?;
                }
            }
        }
    }
    Ok(())
}
fn shuffled(indices: &mut [usize], mut rng: &mut dyn BitGenerator) -> Result<(), SciOxideError> {
    shuffle(indices, &mut rng)
}
fn permutation<S: RealScalar>(
    offsets: &[usize],
    kind: PermutationKind,
    mode: PermutationMode,
    mut rank: usize,
    rng: &mut dyn BitGenerator,
    slot: &mut Slot<S>,
) -> Result<(), SciOxideError> {
    let groups = offsets.len() - 1;
    let n = offsets[1];
    match (kind, mode) {
        (PermutationKind::Independent, PermutationMode::Randomized) => {
            shuffled(&mut slot.indices, rng)?;
            for pair in offsets.windows(2) {
                slot.indices[pair[0]..pair[1]].sort_unstable();
            }
        }
        (PermutationKind::Independent, PermutationMode::Exact) => {
            for (count, pair) in slot.counts.iter_mut().zip(offsets.windows(2)) {
                *count = pair[1] - pair[0];
            }
            for index in 0..slot.indices.len() {
                for g in 0..groups {
                    if slot.counts[g] == 0 {
                        continue;
                    }
                    slot.counts[g] -= 1;
                    let ways = multinomial(&slot.counts)?;
                    if rank < ways {
                        slot.indices[offsets[g + 1] - slot.counts[g] - 1] = index;
                        break;
                    }
                    rank -= ways;
                    slot.counts[g] += 1;
                }
            }
        }
        (PermutationKind::Pairings, _) => {
            for pair in offsets.windows(2) {
                let indices = &mut slot.indices[pair[0]..pair[1]];
                if mode == PermutationMode::Exact {
                    let ways = checked_factorial(n)?;
                    permutation_unrank_into(rank % ways, indices)?;
                    indices.iter_mut().for_each(|i| *i += pair[0]);
                    rank /= ways;
                } else {
                    shuffled(indices, rng)?;
                }
            }
        }
        (PermutationKind::PairedLabels, _) => {
            for row in 0..n {
                for (g, value) in slot.counts.iter_mut().enumerate() {
                    *value = g;
                }
                if mode == PermutationMode::Exact {
                    let ways = checked_factorial(groups)?;
                    permutation_unrank_into(rank % ways, &mut slot.counts)?;
                    rank /= ways;
                } else {
                    shuffled(&mut slot.counts, rng)?;
                }
                for (g, &source) in slot.counts.iter().enumerate() {
                    slot.indices[g * n + row] = source * n + row;
                }
            }
        }
    }
    Ok(())
}
fn power(base: usize, n: usize) -> Result<usize, SciOxideError> {
    (0..n).try_fold(1, |a, _| {
        checked_mul(a, base, scioxide_core::quantity!("permutation_count"))
    })
}
fn multinomial(counts: &[usize]) -> Result<usize, SciOxideError> {
    let mut used = 0;
    let mut ways = 1;
    for &n in counts {
        used = checked_add(used, n, scioxide_core::quantity!("permutation_count"))?;
        ways = checked_mul(
            ways,
            checked_binomial(used, n)?,
            scioxide_core::quantity!("permutation_count"),
        )?;
    }
    Ok(ways)
}
