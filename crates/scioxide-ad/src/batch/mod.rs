//! Sums of a function over many independent items, with and without derivatives.

mod drive;
mod one_shot;
mod reduce;
mod workspace;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;

use scioxide_core::{ExecutionPolicy, InvalidInputReason, RealScalar, SciOxideError};

use crate::{DerivativeMode, KinkPolicy, NonFinitePolicy, Real};

pub use one_shot::{batch_mean, batch_mean_and_grad};
pub use workspace::{BatchRequirements, BatchWorkspace};

/// A function summed over the items of a batch: `Σᵢ f(θ, aᵢ, dᵢ)` or its mean.
///
/// `θ` are the parameters shared by every item, `aᵢ` the active inputs of item
/// `i` and `dᵢ` its passive data. Both methods are generic in the number type, so
/// one implementation serves pricing (`R = Lanes`), forward mode
/// (`R = MultiDual<Lanes, N>`) and reverse mode (`R = Var<Lanes>`), and a single
/// item can run on plain `f64` for testing.
pub trait BatchFunction<S: RealScalar = f64>: Sync {
    /// State that does not depend on the item, built from the parameters.
    type Shared<R: Real<S>>;

    /// Builds the shared state from the parameters, once per block of items.
    ///
    /// # Errors
    ///
    /// Any error of the function; it aborts the batch.
    fn prepare<R: Real<S>>(&self, parameters: &[R]) -> Result<Self::Shared<R>, SciOxideError>;

    /// Evaluates one item (one lane per item when `R` has lanes).
    ///
    /// # Errors
    ///
    /// Any error of the function; it aborts the batch.
    fn item<R: Real<S>>(
        &self,
        shared: &Self::Shared<R>,
        inputs: &[R],
        data: &[R::Base],
    ) -> Result<R, SciOxideError>;
}

/// A source of per-item values: active inputs, their directions or passive data.
///
/// Every item holds `width` values. A batch evaluates `W` consecutive items as
/// the lanes of one pack, so it reads a source one value at a time, for the `W`
/// items of a group: a source that keeps a value of consecutive items in adjacent
/// memory (overlapping windows of one array, one array per value) is copied
/// without rearrangement. A source that can only produce whole rows fills a
/// row-major table first; the table is a source as a slice.
///
/// # Examples
///
/// ```
/// use scioxide_ad::BatchData;
/// use scioxide_core::SciOxideError;
///
/// struct Windows<'a>(&'a [f64]); // item `i` is `x[i..i + width]`
/// impl BatchData for Windows<'_> {
///     fn fill(&self, i: usize, _: usize, k: usize, lanes: &mut [f64]) -> Result<(), SciOxideError> {
///         let end = i + k + lanes.len();
///         let short = SciOxideError::dimension_mismatch(end, self.0.len());
///         lanes.copy_from_slice(self.0.get(i + k..end).ok_or(short)?);
///         Ok(())
///     }
/// }
/// let mut second = [0.0; 3];
/// Windows(&[1.0, 2.0, 3.0, 4.0]).fill(0, 2, 1, &mut second)?;
/// assert_eq!(second, [2.0, 3.0, 4.0]);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait BatchData<S: RealScalar = f64>: Sync {
    /// Writes value `column` of the items `first..first + lanes.len()`, each of
    /// which holds `width` values, to `lanes`.
    ///
    /// A batch calls it once per value and group with `column < width`, at most `W`
    /// lanes and items inside the plan.
    ///
    /// # Errors
    ///
    /// Any error of the source; it aborts the batch.
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [S],
    ) -> Result<(), SciOxideError>;
}

/// A row-major table of items × width: value `column` of item `i` is element
/// `i·width + column`.
impl<S: RealScalar> BatchData<S> for [S] {
    /// # Errors
    ///
    /// `OutOfBounds` if `column ≥ width`, or with the first missing item if the
    /// table holds fewer than `first + lanes.len()` rows.
    #[inline]
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [S],
    ) -> Result<(), SciOxideError> {
        if column >= width {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("batch_columns"),
                column,
                width,
            ));
        }
        let rows = first
            .checked_mul(width)
            .zip(lanes.len().checked_mul(width))
            .and_then(|(start, len)| self.get(start..)?.get(..len));
        let Some(rows) = rows else {
            let count = self.len() / width;
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("batch_rows"),
                first.max(count),
                count,
            ));
        };
        match width {
            1 => lanes.copy_from_slice(rows),
            2 => gather::<S, 2>(rows, column, lanes),
            3 => gather::<S, 3>(rows, column, lanes),
            4 => gather::<S, 4>(rows, column, lanes),
            _ => {
                for (lane, row) in lanes.iter_mut().zip(rows.chunks_exact(width)) {
                    *lane = row[column];
                }
            }
        }
        Ok(())
    }
}

/// A row-major table in a `Vec`, read as its slice.
impl<S: RealScalar> BatchData<S> for Vec<S> {
    /// # Errors
    ///
    /// Those of the slice source.
    #[inline]
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [S],
    ) -> Result<(), SciOxideError> {
        <[S] as BatchData<S>>::fill(self.as_slice(), first, width, column, lanes)
    }
}

/// A row-major table in an array, read as its slice; `&[]` is an empty source.
impl<S: RealScalar, const N: usize> BatchData<S> for [S; N] {
    /// # Errors
    ///
    /// Those of the slice source.
    #[inline]
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [S],
    ) -> Result<(), SciOxideError> {
        <[S] as BatchData<S>>::fill(self.as_slice(), first, width, column, lanes)
    }
}

/// A closure as a [`BatchData`] source.
///
/// The closure has the arguments of [`BatchData::fill`]: it writes value `column`
/// of the items `first..first + lanes.len()`, each of which holds `width` values,
/// to `lanes`. A source that draws values on demand (for example one random stream
/// per item) needs no table; a source that can only produce whole rows fills a
/// row-major table first. The adapter checks and allocates nothing.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{BatchData, BatchDataFn};
/// // Value `column` of item `i` is `i + column / 10`.
/// let source = BatchDataFn::new(|first, _, column, lanes: &mut [f64]| {
///     for (k, lane) in lanes.iter_mut().enumerate() {
///         *lane = (first + k) as f64 + column as f64 / 10.0;
///     }
///     Ok(())
/// });
/// let mut lanes = [0.0; 3];
/// source.fill(4, 2, 1, &mut lanes)?;
/// assert_eq!(lanes, [4.1, 5.1, 6.1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct BatchDataFn<S, F> {
    fill: F,
    scalar: PhantomData<fn() -> S>,
}

impl<S, F> core::fmt::Debug for BatchDataFn<S, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchDataFn").finish_non_exhaustive()
    }
}

impl<S, F> BatchDataFn<S, F>
where
    S: RealScalar,
    F: Fn(usize, usize, usize, &mut [S]) -> Result<(), SciOxideError> + Sync,
{
    /// Wraps `fill(first, width, column, lanes)`.
    pub const fn new(fill: F) -> Self {
        Self {
            fill,
            scalar: PhantomData,
        }
    }
}

impl<S, F> BatchData<S> for BatchDataFn<S, F>
where
    S: RealScalar,
    F: Fn(usize, usize, usize, &mut [S]) -> Result<(), SciOxideError> + Sync,
{
    /// # Errors
    ///
    /// Those of the closure, unchanged.
    #[inline]
    fn fill(
        &self,
        first: usize,
        width: usize,
        column: usize,
        lanes: &mut [S],
    ) -> Result<(), SciOxideError> {
        (self.fill)(first, width, column, lanes)
    }
}

/// Writes element `column` of each row of `K` values to `lanes`; the constant
/// width gives a constant stride.
#[inline(always)]
fn gather<S: Copy, const K: usize>(rows: &[S], column: usize, lanes: &mut [S]) {
    let (rows, _) = rows.as_chunks::<K>();
    for (lane, row) in lanes.iter_mut().zip(rows) {
        *lane = row[column];
    }
}

/// How the items of a batch are reduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchReduction {
    /// The sum over items; `+0` for no items.
    #[default]
    Sum,
    /// The mean over items.
    Mean,
}

/// The number type a batch evaluation used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchMode {
    /// Plain lanes, without derivatives.
    Value,
    /// Forward mode with `tangents` tangents per lane.
    Forward {
        /// Tangents carried per lane.
        tangents: usize,
    },
    /// Reverse mode on a tape per worker.
    Reverse,
}

/// Options of a [`BatchPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchOptions {
    policy: ExecutionPolicy,
    block: NonZeroUsize,
    mode: DerivativeMode,
    non_finite: NonFinitePolicy,
    kink: KinkPolicy,
    byte_limit: Option<usize>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchOptions {
    /// Returns the defaults: sequential execution, blocks of 256 items,
    /// [`DerivativeMode::Auto`], [`NonFinitePolicy::Propagate`],
    /// [`KinkPolicy::Midpoint`] and no byte limit.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
            block: match NonZeroUsize::new(256) {
                Some(block) => block,
                None => NonZeroUsize::MIN,
            },
            mode: DerivativeMode::Auto,
            non_finite: NonFinitePolicy::Propagate,
            kink: KinkPolicy::Midpoint,
            byte_limit: None,
        }
    }
    /// Sets how blocks are distributed over workers.
    #[must_use]
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the items per block, rounded up to a multiple of the lane width.
    #[must_use]
    pub const fn with_block(mut self, block: NonZeroUsize) -> Self {
        self.block = block;
        self
    }
    /// Sets how gradients are computed.
    #[must_use]
    pub const fn with_derivative_mode(mut self, mode: DerivativeMode) -> Self {
        self.mode = mode;
        self
    }
    /// Sets the treatment of non-finite parameters, inputs and results.
    #[must_use]
    pub const fn with_non_finite_policy(mut self, policy: NonFinitePolicy) -> Self {
        self.non_finite = policy;
        self
    }
    /// Sets what a kink hit does; [`KinkPolicy::Error`] forces reverse mode.
    ///
    /// The policy applies to the tapes of reverse mode, in
    /// [`BatchPlan::evaluate_into`] and [`BatchPlan::hvp_into`]. Value mode and
    /// [`BatchPlan::jvp_into`] record no tape and count no kinks.
    #[must_use]
    pub const fn with_kink_policy(mut self, policy: KinkPolicy) -> Self {
        self.kink = policy;
        self
    }
    /// Limits the bytes each tape of a worker may reserve; a worker has a tape for
    /// gradients and one for Hessian-vector products, each limited separately. The
    /// limit applies to the growth of the tapes when this plan evaluates: a
    /// workspace whose tapes grew under a larger limit keeps their storage.
    #[must_use]
    pub const fn with_byte_limit(mut self, limit: Option<usize>) -> Self {
        self.byte_limit = limit;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the requested items per block.
    #[must_use]
    pub const fn block(&self) -> NonZeroUsize {
        self.block
    }
    /// Returns the derivative mode.
    #[must_use]
    pub const fn derivative_mode(&self) -> DerivativeMode {
        self.mode
    }
    /// Returns the non-finite policy.
    #[must_use]
    pub const fn non_finite_policy(&self) -> NonFinitePolicy {
        self.non_finite
    }
    /// Returns the kink policy.
    #[must_use]
    pub const fn kink_policy(&self) -> KinkPolicy {
        self.kink
    }
    /// Returns the byte limit of each worker's tape.
    #[must_use]
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
}

/// The buffers a batch evaluation writes. Requesting a gradient is the only
/// difference between evaluating with and without AD.
#[derive(Debug, Default)]
pub struct BatchOutputs<'a, S> {
    pub(crate) values: Option<&'a mut [S]>,
    pub(crate) parameter_gradient: Option<&'a mut [S]>,
    pub(crate) parameter_gradient_error: Option<&'a mut [S]>,
    pub(crate) input_gradient: Option<&'a mut [S]>,
}

impl<'a, S> BatchOutputs<'a, S> {
    /// Requests nothing but the reduced value.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            values: None,
            parameter_gradient: None,
            parameter_gradient_error: None,
            input_gradient: None,
        }
    }
    /// Writes `f(θ, aᵢ, dᵢ)` of every item (not scaled by the reduction).
    #[must_use]
    pub fn with_values(mut self, values: &'a mut [S]) -> Self {
        self.values = Some(values);
        self
    }
    /// Writes the gradient of the reduced value with respect to the parameters.
    #[must_use]
    pub fn with_parameter_gradient(mut self, gradient: &'a mut [S]) -> Self {
        self.parameter_gradient = Some(gradient);
        self
    }
    /// Writes the standard error of each entry of the parameter gradient, by batch
    /// means over blocks.
    #[must_use]
    pub fn with_parameter_gradient_error(mut self, error: &'a mut [S]) -> Self {
        self.parameter_gradient_error = Some(error);
        self
    }
    /// Writes the gradient of each item with respect to its inputs (row-major items
    /// × inputs, not scaled by the reduction).
    #[must_use]
    pub fn with_input_gradient(mut self, gradient: &'a mut [S]) -> Self {
        self.input_gradient = Some(gradient);
        self
    }

    pub(crate) fn wants_gradient(&self) -> bool {
        self.parameter_gradient.is_some()
            || self.parameter_gradient_error.is_some()
            || self.input_gradient.is_some()
    }
}

/// The result of a batch evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchEstimate<S> {
    pub(crate) value: S,
    pub(crate) standard_error: Option<S>,
    pub(crate) derivative: Option<S>,
    pub(crate) items: usize,
    pub(crate) blocks: usize,
    pub(crate) mode: BatchMode,
    pub(crate) kinks: Option<u64>,
}

impl<S: Copy> BatchEstimate<S> {
    /// Returns the sum or mean over items.
    #[must_use]
    pub const fn value(&self) -> S {
        self.value
    }
    /// Returns the standard error of the mean, `s/√n`, for [`BatchReduction::Mean`]
    /// with at least two items.
    #[must_use]
    pub const fn standard_error(&self) -> Option<S> {
        self.standard_error
    }
    /// Returns the reduced directional derivative of [`BatchPlan::jvp_into`].
    #[must_use]
    pub const fn derivative(&self) -> Option<S> {
        self.derivative
    }
    /// Returns the number of items.
    #[must_use]
    pub const fn items(&self) -> usize {
        self.items
    }
    /// Returns the number of blocks.
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }
    /// Returns the number type used.
    #[must_use]
    pub const fn mode(&self) -> BatchMode {
        self.mode
    }
    /// Returns the kink hits in reverse mode: one per operation and item lane in
    /// which an active argument sits on a kink, plus those of `prepare` once per
    /// block.
    #[must_use]
    pub const fn kinks(&self) -> Option<u64> {
        self.kinks
    }
}

/// A batch of `items` independent items evaluated `W` lanes at a time.
///
/// Items are split into blocks and blocks into groups of `W` consecutive items.
/// Each block calls [`BatchFunction::prepare`] once and, per group,
/// [`BatchData::fill`] once per value and [`BatchFunction::item`] once, except
/// that in reverse mode the first block of a worker whose tape must grow runs
/// twice, as every recording that does not fit; missing lanes of the last group
/// repeat its last item and are excluded from every output. Blocks are distributed over workers by the execution policy,
/// and the caller thread adds the block results by pairwise summation over the
/// block index, so estimates, gradients and errors are identical for every
/// execution policy and worker count. The value has the same bits in value,
/// forward and reverse mode.
///
/// Without a gradient the items run on [`Lanes<S, W>`](crate::Lanes). With one,
/// forward mode (`MultiDual<Lanes<S, W>, N>` with one tangent per parameter and
/// input) runs when parameters plus inputs are at most four, and reverse mode
/// (`Var<Lanes<S, W>>`, one tape per worker, a shared prefix recorded once per
/// block) otherwise.
///
/// # Accuracy
///
/// Items and their derivatives have the accuracy of the number types (see the
/// [crate documentation](crate#accuracy)). A sum over `n` items in blocks of `b`
/// adds recursively within blocks and pairwise over the `B` blocks, so its error is
/// at most `γₖ·Σᵢ |yᵢ|` with `k = b + ⌈log₂ B⌉` and `γₖ = kε/(1 − kε)`; gradients
/// are sums of the same form.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{BatchFunction, BatchOptions, BatchOutputs, BatchPlan, BatchReduction, Real};
/// use scioxide_core::{ExecutionContext, SciOxideError};
///
/// /// The mean of `θ·zᵢ²` over draws `zᵢ`.
/// struct Scaled;
/// impl BatchFunction for Scaled {
///     type Shared<R: Real> = R;
///     fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> { Ok(p[0]) }
///     fn item<R: Real>(&self, theta: &R, _: &[R], z: &[R::Base]) -> Result<R, SciOxideError> {
///         Ok(*theta * R::passive(z[0]) * R::passive(z[0]))
///     }
/// }
/// let draws = [1.0, 2.0, 3.0];
/// let plan = BatchPlan::<4>::new(3, 1, 0, 1, BatchReduction::Mean, BatchOptions::new())?;
/// let mut workspace = plan.workspace(1)?;
/// let mut gradient = [0.0];
/// let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
/// let estimate = plan.evaluate_into(&Scaled, &[2.0], &[][..], &draws[..], outputs, &mut workspace, ExecutionContext::sequential())?;
/// assert_eq!((estimate.value(), gradient[0]), (28.0 / 3.0, 14.0 / 3.0));
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan<const W: usize = 8> {
    items: usize,
    parameters: usize,
    inputs: usize,
    data: usize,
    reduction: BatchReduction,
    options: BatchOptions,
    block: usize,
}

impl<const W: usize> BatchPlan<W> {
    /// Builds a plan for `items` items with `parameters` shared parameters,
    /// `inputs` active inputs and `data` passive values per item.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `W = 0`, or for [`BatchReduction::Mean`] over zero items.
    /// - `SizeOverflow` if the rounded block size overflows.
    pub fn new(
        items: usize,
        parameters: usize,
        inputs: usize,
        data: usize,
        reduction: BatchReduction,
        options: BatchOptions,
    ) -> Result<Self, SciOxideError> {
        if W == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("lanes"),
                InvalidInputReason::OutOfRange,
                "a batch needs at least one lane",
            ));
        }
        if reduction == BatchReduction::Mean && items == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("items"),
                InvalidInputReason::Empty,
                "the mean over zero items is undefined",
            ));
        }
        let block = options
            .block
            .get()
            .checked_next_multiple_of(W)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("batch_block")))?;
        Ok(Self {
            items,
            parameters,
            inputs,
            data,
            reduction,
            options,
            block,
        })
    }

    /// Returns the number of items.
    #[must_use]
    pub const fn items(&self) -> usize {
        self.items
    }
    /// Returns the number of parameters.
    #[must_use]
    pub const fn parameters(&self) -> usize {
        self.parameters
    }
    /// Returns the number of active inputs per item.
    #[must_use]
    pub const fn inputs(&self) -> usize {
        self.inputs
    }
    /// Returns the number of passive data values per item.
    #[must_use]
    pub const fn data(&self) -> usize {
        self.data
    }
    /// Returns the reduction.
    #[must_use]
    pub const fn reduction(&self) -> BatchReduction {
        self.reduction
    }
    /// Returns the options.
    #[must_use]
    pub const fn options(&self) -> &BatchOptions {
        &self.options
    }
    /// Returns the items per block (a multiple of `W`).
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
    }
    /// Returns the number of blocks.
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.items.div_ceil(self.block)
    }

    /// Returns the storage a workspace for `workers` workers needs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `workers`, `OutOfRange`) unless `1 ≤ workers ≤ 64`.
    /// - `SizeOverflow` if a size overflows.
    pub fn workspace_requirements(
        &self,
        workers: usize,
    ) -> Result<BatchRequirements, SciOxideError> {
        BatchRequirements::for_plan(self, workers)
    }

    /// Returns a workspace reserved for `workers` workers.
    ///
    /// # Errors
    ///
    /// The errors of [`BatchPlan::workspace_requirements`] and `AllocationFailure`.
    pub fn workspace<S: RealScalar>(
        &self,
        workers: usize,
    ) -> Result<BatchWorkspace<S, W>, SciOxideError> {
        let requirements = self.workspace_requirements(workers)?;
        let mut workspace = BatchWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
}
