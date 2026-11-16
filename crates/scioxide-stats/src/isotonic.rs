//! Weighted isotonic regression by the pool-adjacent-violators algorithm in O(n) time.
mod native;

use crate::NonFinitePolicy;
use crate::arithmetic::promote;
use crate::arithmetic::{Arithmetic, Compensated};
use crate::error_helpers::invalid;
use crate::ranking::finite_cmp;
use alloc::vec::Vec;
use core::cmp::Ordering;

use scioxide_core::backend::try_filled;
use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};

/// Order constraint of an isotonic fit, in the order of the observations.
///
/// Both directions are non-strict: equal adjacent fitted values are allowed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IsotonicDirection {
    /// Fitted values never decrease.
    #[default]
    Increasing,
    /// Fitted values never increase.
    Decreasing,
}

/// Options for one slice. Zero-weight observations are skipped before their
/// values are checked. A skipped position receives the fit of the preceding
/// position, or of the first fitted position for leading gaps. A nonempty input
/// needs positive total weight.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawIsotonicOptions {
    /// Order constraint; increasing by default.
    pub direction: IsotonicDirection,
    /// Rejects (default) or skips non-finite observations with positive weight.
    pub non_finite: NonFinitePolicy,
}

/// Maximal run of positions `start..end` on which an isotonic fit is constant.
///
/// The positions are indices into the fitted slice or lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicBlock<S> {
    /// First position of the block.
    pub start: usize,
    /// One past the last position of the block.
    pub end: usize,
    /// Fitted value on the block: the weighted mean of its observations. A zero is `+0`.
    pub value: S,
    /// Total weight of the block, or `None` if it exceeds the largest finite `S`.
    pub weight: Option<S>,
}
impl<S: RealScalar> Default for IsotonicBlock<S> {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            value: S::ZERO,
            weight: Some(S::ZERO),
        }
    }
}

/// Summary of an isotonic fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicDiagnostics<S> {
    /// Number of maximal constant blocks in the fit, after rounding to `S`.
    pub block_count: usize,
    /// Number of observations used in the fit, those with positive weight.
    pub retained: usize,
    /// Number of skipped observations: zero-weight, missing or non-finite.
    pub excluded: usize,
    /// Total weight of the fitted observations, or `None` if it exceeds the
    /// largest finite `S`.
    pub weight: Option<S>,
    /// Weighted residual sum of squares `Σ wᵢ (yᵢ − f̂ᵢ)²` at the returned fit, or
    /// `None` if it exceeds the largest finite `S`. Underflow rounds to zero.
    pub squared_error: Option<S>,
}
impl<S: RealScalar> Default for IsotonicDiagnostics<S> {
    fn default() -> Self {
        Self {
            block_count: 0,
            retained: 0,
            excluded: 0,
            weight: Some(S::ZERO),
            squared_error: Some(S::ZERO),
        }
    }
}

/// Weighted isotonic regression fit, returned by
/// [`isotonic_regression_with_options`](crate::isotonic_regression_with_options).
///
/// For observations `y₁, …, yₙ` with weights `wᵢ ≥ 0`, the fit `f̂` minimizes
///
/// ```text
/// Σᵢ wᵢ (yᵢ − fᵢ)²   subject to   f₁ ≤ f₂ ≤ … ≤ fₙ
/// ```
///
/// (or `f₁ ≥ … ≥ fₙ` for [`IsotonicDirection::Decreasing`]). The solution is
/// piecewise constant, and each block takes the weighted mean of its
/// observations. The pool-adjacent-violators algorithm computes it in O(n)
/// time and memory.
///
/// # References
///
/// R. E. Barlow, D. J. Bartholomew, J. M. Bremner and H. D. Brunk,
/// *Statistical Inference under Order Restrictions*, Wiley, 1972.
#[derive(Clone, Debug)]
pub struct IsotonicResult<S> {
    /// Fitted values in observation order.
    fitted: Vec<S>,
    /// Maximal constant blocks covering every position.
    blocks: Vec<IsotonicBlock<S>>,
    /// Summary of the fit.
    diagnostics: IsotonicDiagnostics<S>,
}
impl<S> IsotonicResult<S> {
    pub(crate) fn from_fit(
        fitted: Vec<S>,
        blocks: Vec<IsotonicBlock<S>>,
        mut diagnostics: IsotonicDiagnostics<S>,
    ) -> Self {
        diagnostics.block_count = blocks.len();
        Self {
            fitted,
            blocks,
            diagnostics,
        }
    }
    /// Returns the fitted values in observation order.
    pub fn fitted(&self) -> &[S] {
        &self.fitted
    }
    /// Returns the maximal constant blocks, in order, covering every position.
    pub fn blocks(&self) -> &[IsotonicBlock<S>] {
        &self.blocks
    }
    /// Returns the summary of the fit. Its `block_count` equals `blocks().len()`.
    pub const fn diagnostics(&self) -> IsotonicDiagnostics<S>
    where
        S: Copy,
    {
        self.diagnostics
    }
    /// Returns the fitted values, blocks and summary by value.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (Vec<S>, Vec<IsotonicBlock<S>>, IsotonicDiagnostics<S>) {
        (self.fitted, self.blocks, self.diagnostics)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    value: f64,
    weight: f64,
}
// Source exponents in +/-64 keep weighted sums, residual squares and every
// usize-count reduction in normal binary64. Cancellation of source products
// can reach exponent -232; division by the largest mass and a subsequent
// weighted square still stay above -1022. Wider inputs retain scaled state.
fn native(entry: Entry) -> bool {
    crate::arithmetic::native(entry.value, 64) && crate::arithmetic::native(entry.weight, 64)
}

/// Pool state for inputs with binary exponents beyond `±64`. A pool keeps
/// compensated `f64` sums ([`native::Sums`]) until a member outside that range
/// joins, then converts both sums exactly to scaled form.
#[derive(Clone, Copy)]
enum PoolState {
    Native {
        sums: native::Sums,
        mean: f64,
    },
    Extended {
        mass: ScaledAccumulator<f64>,
        sum: ScaledAccumulator<f64>,
        mean: Scaled<f64>,
    },
}
impl PoolState {
    fn extended(self) -> Result<(ScaledAccumulator<f64>, ScaledAccumulator<f64>), SciOxideError> {
        match self {
            Self::Native { sums, .. } => Ok((promote(sums.mass())?, promote(sums.sum())?)),
            Self::Extended { mass, sum, .. } => Ok((mass, sum)),
        }
    }
    fn mean(self) -> Result<Scaled<f64>, SciOxideError> {
        match self {
            Self::Native { mean, .. } => Scaled::from_value(mean),
            Self::Extended { mean, .. } => Ok(mean),
        }
    }
    #[inline]
    fn rounded(self) -> Result<f64, SciOxideError> {
        match self {
            Self::Native { mean, .. } => Ok(mean),
            Self::Extended { mean, .. } => mean.round(),
        }
    }
    #[inline]
    fn compare(self, right: Self) -> Result<Ordering, SciOxideError> {
        if let (Self::Native { mean: a, .. }, Self::Native { mean: b, .. }) = (self, right) {
            Ok(finite_cmp(&a, &b))
        } else {
            self.compare_extended(right)
        }
    }
    #[cold]
    fn compare_extended(self, right: Self) -> Result<Ordering, SciOxideError> {
        Ok(self.mean()?.cmp_value(right.mean()?))
    }
    #[cold]
    fn single_extended(entry: Entry) -> Result<Self, SciOxideError> {
        let mass = Scaled::from_value(entry.weight)?;
        let mean = Scaled::from_value(entry.value)?;
        Ok(Self::Extended {
            mass: ScaledAccumulator::from_scaled(mass),
            sum: ScaledAccumulator::from_scaled(mass.mul(mean)?),
            mean,
        })
    }
    #[cold]
    fn merge_extended(self, right: Self, low: f64, high: f64) -> Result<Self, SciOxideError> {
        let (mut mass, mut sum) = self.extended()?;
        let (rm, rs) = right.extended()?;
        mass.merge(rm)?;
        sum.merge(rs)?;
        let mut mean = sum.total()?.div(mass.total()?)?;
        // Clamp the convex mean before projection, including at MAX.
        let lower = Scaled::from_value(low)?;
        let upper = Scaled::from_value(high)?;
        if mean.cmp_value(lower) == Ordering::Less {
            mean = lower;
        }
        if mean.cmp_value(upper) == Ordering::Greater {
            mean = upper;
        }
        Ok(Self::Extended { mass, sum, mean })
    }
}

#[derive(Clone, Copy)]
struct Pool {
    start: usize,
    state: PoolState,
    low: f64,
    high: f64,
}
impl Pool {
    fn empty() -> Self {
        Self {
            start: 0,
            state: PoolState::Native {
                sums: native::Sums::ZERO,
                mean: 0.0,
            },
            low: 0.0,
            high: 0.0,
        }
    }
    #[inline]
    fn single(start: usize, entry: Entry) -> Result<Self, SciOxideError> {
        let state = if native(entry) {
            PoolState::Native {
                sums: native::Sums::single(entry),
                mean: entry.value,
            }
        } else {
            PoolState::single_extended(entry)?
        };
        Ok(Self {
            start,
            state,
            low: entry.value,
            high: entry.value,
        })
    }
    #[inline]
    fn merge(self, right: Self) -> Result<Self, SciOxideError> {
        let low = self.low.min(right.low);
        let high = self.high.max(right.high);
        let state =
            if let (PoolState::Native { sums: left, .. }, PoolState::Native { sums: right, .. }) =
                (self.state, right.state)
            {
                let sums = left.combined(right);
                PoolState::Native {
                    sums,
                    mean: sums.mean(low, high),
                }
            } else {
                self.state.merge_extended(right.state, low, high)?
            };
        Ok(Self {
            start: self.start,
            state,
            low,
            high,
        })
    }
}

fn diagnostic<S: RealScalar>(value: Scaled<f64>) -> Result<Option<S>, SciOxideError> {
    if value.cmp_value(Scaled::from_value(S::MAX.try_to_f64_exact()?)?) == Ordering::Greater {
        return Ok(None);
    }
    Ok(Some(S::try_from_f64_lossy(value.round()?)?))
}

/// O(n) storage for the observations, the pool stack, the fit and the blocks.
/// Fitting never grows it. Pools use compensated `f64` sums, or scaled sums on a
/// separate stack when an input has a binary exponent beyond `±64`.
pub(crate) struct RawIsotonicWorkspace<S: RealScalar> {
    entries: Vec<Entry>,
    native_pools: Vec<native::Pool>,
    pools: Vec<Pool>,
    pub(crate) fitted: Vec<S>,
    pub(crate) blocks: Vec<IsotonicBlock<S>>,
}
impl<S: RealScalar> Default for RawIsotonicWorkspace<S> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            native_pools: Vec::new(),
            pools: Vec::new(),
            fitted: Vec::new(),
            blocks: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawIsotonicWorkspace<S> {
    pub(crate) fn required_bytes(n: usize) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[
            (n, core::mem::size_of::<Entry>()),
            (n, core::mem::size_of::<native::Pool>()),
            (n, core::mem::size_of::<Pool>()),
            (n, core::mem::size_of::<S>()),
            (n, core::mem::size_of::<IsotonicBlock<S>>()),
        ])
    }
    /// Returns the largest number of observations the workspace can fit.
    pub(crate) fn capacity(&self) -> usize {
        self.entries.len()
    }
    /// Grows the workspace to fit `n` observations, keeping storage that is large enough.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure; the previous storage is then unchanged.
    pub(crate) fn reserve(&mut self, n: usize) -> Result<(), SciOxideError> {
        if self.capacity() >= n {
            return Ok(());
        }
        let next = Self {
            entries: try_filled(
                n,
                Entry {
                    value: 0.0,
                    weight: 0.0,
                },
                scioxide_core::quantity!("isotonic_entries"),
            )?,
            native_pools: try_filled(
                n,
                native::Pool::EMPTY,
                scioxide_core::quantity!("isotonic_native_pools"),
            )?,
            pools: try_filled(n, Pool::empty(), scioxide_core::quantity!("isotonic_pools"))?,
            fitted: try_filled(n, S::ZERO, scioxide_core::quantity!("isotonic_fitted"))?,
            blocks: try_filled(
                n,
                IsotonicBlock::default(),
                scioxide_core::quantity!("isotonic_blocks"),
            )?,
        };
        *self = next;
        Ok(())
    }
    /// Checks and stores one observation. Returns whether its value and weight
    /// have binary exponents within `±64`; skipped observations always return `true`.
    pub(crate) fn admit(
        &mut self,
        index: usize,
        value: S,
        weight: S,
        policy: NonFinitePolicy,
        argument: scioxide_core::Argument,
    ) -> Result<bool, SciOxideError> {
        if !weight.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "nonfinite weight",
            ));
        }
        if weight < S::ZERO {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "negative weight",
            ));
        }
        self.entries[index] = Entry {
            value: 0.0,
            weight: 0.0,
        };
        if weight == S::ZERO {
            return Ok(true);
        }
        if !value.is_finite() {
            return if policy == NonFinitePolicy::Skip {
                Ok(true)
            } else {
                Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "nonfinite observation",
                ))
            };
        }
        let entry = Entry {
            value: value.try_to_f64_exact()?,
            weight: weight.try_to_f64_exact()?,
        };
        self.entries[index] = entry;
        Ok(native(entry))
    }
    pub(crate) fn fit(
        &mut self,
        n: usize,
        direction: IsotonicDirection,
        ordinary: bool,
        argument: scioxide_core::Argument,
    ) -> Result<IsotonicDiagnostics<S>, SciOxideError> {
        if ordinary {
            return match direction {
                IsotonicDirection::Increasing => native::fit::<S, true>(
                    &self.entries[..n],
                    &mut self.native_pools,
                    &mut self.fitted[..n],
                    &mut self.blocks,
                    argument,
                ),
                IsotonicDirection::Decreasing => native::fit::<S, false>(
                    &self.entries[..n],
                    &mut self.native_pools,
                    &mut self.fitted[..n],
                    &mut self.blocks,
                    argument,
                ),
            };
        }
        self.fit_extended(n, direction, argument)
    }

    #[cold]
    fn fit_extended(
        &mut self,
        n: usize,
        direction: IsotonicDirection,
        argument: scioxide_core::Argument,
    ) -> Result<IsotonicDiagnostics<S>, SciOxideError> {
        let mut count = 0;
        let mut retained = 0;
        for (index, &entry) in self.entries[..n].iter().enumerate() {
            if entry.weight == 0.0 {
                continue;
            }
            retained += 1;
            // Keep the incoming/merged pool in a register candidate. Publish to
            // the stack only after all violations are resolved, avoiding writes
            // and reloads of the large extended-state slot on every merge.
            let mut pool = Pool::single(if count == 0 { 0 } else { index }, entry)?;
            while count > 0 {
                let left = self.pools[count - 1];
                let order = left.state.compare(pool.state)?;
                let ordered = match direction {
                    IsotonicDirection::Increasing => order == Ordering::Less,
                    IsotonicDirection::Decreasing => order == Ordering::Greater,
                };
                if ordered {
                    break;
                }
                pool = left.merge(pool)?;
                count -= 1;
            }
            self.pools[count] = pool;
            count += 1;
        }
        if n > 0 && count == 0 {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::OutOfRange,
                "nonempty lane has no positive retained mass",
            ));
        }
        for pool in 0..count {
            let start = self.pools[pool].start;
            let end = if pool + 1 == count {
                n
            } else {
                self.pools[pool + 1].start
            };
            let value = S::try_from_f64_lossy(self.pools[pool].state.rounded()?)?;
            self.fitted[start..end].fill(if value == S::ZERO { S::ZERO } else { value });
        }
        self.summarize::<Scaled<f64>>(
            n,
            retained,
            count,
            |state| match state {
                PoolState::Native { sums, .. } => crate::arithmetic::encode_components(sums.mass()),
                PoolState::Extended { mass, .. } => Ok(mass.parts()),
            },
            diagnostic,
        )
    }

    fn summarize<N: Arithmetic>(
        &mut self,
        n: usize,
        retained: usize,
        count: usize,
        pool_mass: impl Fn(PoolState) -> Result<[N; 2], SciOxideError>,
        project: impl Fn(N) -> Result<Option<S>, SciOxideError>,
    ) -> Result<IsotonicDiagnostics<S>, SciOxideError> {
        let mut mass = Compensated::<N>::zero();
        let mut loss = Compensated::<N>::zero();
        let mut block_mass = Compensated::<N>::zero();
        let mut start = 0;
        let mut block_count = 0;
        // The PAVA stack already owns compensated masses. Reuse both parts,
        // merging adjacent pools only when their published values round equal.
        for (index, pool) in self.pools[..count].iter().enumerate() {
            for part in pool_mass(pool.state)? {
                mass = mass.added(part)?;
                block_mass = block_mass.added(part)?;
            }
            let end = if index + 1 == count {
                n
            } else {
                self.pools[index + 1].start
            };
            let fit = self.fitted[pool.start];
            if end == n || fit != self.fitted[end] {
                self.blocks[block_count] = IsotonicBlock {
                    start,
                    end,
                    value: fit,
                    weight: project(block_mass.total()?)?,
                };
                block_count += 1;
                start = end;
                block_mass = Compensated::zero();
            }
        }
        for (&fit, entry) in self.fitted[..n].iter().zip(&self.entries[..n]) {
            if entry.weight > 0.0 {
                let residual = N::number(entry.value)?.sub(N::number(fit.try_to_f64_exact()?)?)?;
                loss = loss.added(N::number(entry.weight)?.mul(residual.square()?)?)?;
            }
        }
        Ok(IsotonicDiagnostics {
            block_count,
            retained,
            excluded: n - retained,
            weight: project(mass.total()?)?,
            squared_error: project(loss.total()?)?,
        })
    }
}

/// Fits one slice into `output` and `blocks`. `output.len()` must equal
/// `observations.len()`, and `blocks` needs at least that many slots. Only the
/// first `block_count` blocks are written. O(n) time.
///
/// # Errors
/// Fails on inconsistent lengths, a workspace that is too small, a negative or
/// non-finite weight, a rejected observation, zero total weight for a nonempty
/// input, or a numerical failure.
pub(crate) fn isotonic_regression_into<S: RealScalar>(
    observations: &[S],
    weights: Option<&[S]>,
    options: RawIsotonicOptions,
    workspace: &mut RawIsotonicWorkspace<S>,
    output: &mut [S],
    blocks: &mut [IsotonicBlock<S>],
) -> Result<IsotonicDiagnostics<S>, SciOxideError> {
    let n = observations.len();
    if output.len() != n || weights.is_some_and(|w| w.len() != n) || blocks.len() < n {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "slice/output shape mismatch",
        ));
    }
    if workspace.capacity() < n {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient isotonic workspace",
        ));
    }
    let mut ordinary = true;
    for (index, &value) in observations.iter().enumerate() {
        ordinary &= workspace.admit(
            index,
            value,
            weights.map_or(S::ONE, |w| w[index]),
            options.non_finite,
            scioxide_core::argument!("weights"),
        )?;
    }
    let diagnostics = workspace.fit(
        n,
        options.direction,
        ordinary,
        scioxide_core::argument!("weights"),
    )?;
    output.copy_from_slice(&workspace.fitted[..n]);
    blocks[..diagnostics.block_count].copy_from_slice(&workspace.blocks[..diagnostics.block_count]);
    Ok(diagnostics)
}
