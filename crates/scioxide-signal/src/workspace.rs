//! Signal workspaces, workspace requirements, execution options and call shapes.
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{Capacity, Complex, ExecutionPolicy, NumericScalar, SciOxideError};

/// Scratch storage that one signal operation needs, with an optional byte limit.
///
/// The requirements count elements of the three buffers of a
/// [`SignalWorkspace<T>`]: sample elements `T`, real elements `T::Real` and
/// complex elements `Complex<T::Real>`. Plans report requirements without a
/// byte limit. You can set one with [`with_max_bytes`](Self::with_max_bytes);
/// [`SignalWorkspace::try_reserve`] and [`SignalWorkspace::check`] enforce it,
/// and execution ignores it. Requirements of several operations combine with
/// [`max`](Self::max), so one workspace reserved for the largest call serves
/// every smaller one. See [Reusing plans and
/// workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_core::{Capacity, SciOxideErrorKind};
/// use scioxide_signal::{FourierResamplePlan, SignalExecutionOptions, SignalWorkspace};
/// let plan = FourierResamplePlan::<f64>::new(&[8], 0, 16, SignalExecutionOptions::new())?;
/// let requirements = plan.workspace_requirements(())?;
/// let mut workspace = SignalWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// // A byte limit below the reserved capacity is a resource limit.
/// let limited = requirements.with_max_bytes(1);
/// assert_eq!(
///     workspace.check(&limited).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct SignalRequirements {
    values: usize,
    real: usize,
    complex: usize,
    max_bytes: usize,
}

impl SignalRequirements {
    /// Unlimited requirements for the three element classes.
    pub(crate) const fn new(values: usize, real: usize, complex: usize) -> Self {
        Self {
            values,
            real,
            complex,
            max_bytes: usize::MAX,
        }
    }

    /// Sample-type element count only.
    pub(crate) const fn values(values: usize) -> Self {
        Self::new(values, 0, 0)
    }

    /// Complex element count only.
    pub(crate) const fn complex(complex: usize) -> Self {
        Self::new(0, 0, complex)
    }

    /// Returns the elementwise maximum of the element counts and the smaller of
    /// the two byte limits.
    pub const fn max(self, other: Self) -> Self {
        Self {
            values: if self.values > other.values {
                self.values
            } else {
                other.values
            },
            real: if self.real > other.real {
                self.real
            } else {
                other.real
            },
            complex: if self.complex > other.complex {
                self.complex
            } else {
                other.complex
            },
            max_bytes: if self.max_bytes < other.max_bytes {
                self.max_bytes
            } else {
                other.max_bytes
            },
        }
    }

    /// Returns the number of sample elements `T`.
    pub const fn values_len(self) -> usize {
        self.values
    }

    /// Returns the number of real elements `T::Real`.
    pub const fn real_len(self) -> usize {
        self.real
    }

    /// Returns the number of complex elements `Complex<T::Real>`.
    pub const fn complex_len(self) -> usize {
        self.complex
    }

    /// Returns the byte limit, or `usize::MAX` when there is none.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }

    /// Sets the largest number of bytes a workspace may hold when it is reserved
    /// or checked against these requirements.
    pub const fn with_max_bytes(mut self, limit: usize) -> Self {
        self.max_bytes = limit;
        self
    }

    /// Returns the number of bytes the three buffers need for sample type `T`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn bytes<T: NumericScalar>(self) -> Result<usize, SciOxideError> {
        bytes::<T>(self.values, self.real, self.complex)
    }

    fn limit(self, bytes: usize) -> Result<(), SciOxideError> {
        if bytes > self.max_bytes {
            Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("signal_workspace_bytes"),
                bytes,
                self.max_bytes,
            ))
        } else {
            Ok(())
        }
    }
}

fn bytes<T: NumericScalar>(
    values: usize,
    real: usize,
    complex: usize,
) -> Result<usize, SciOxideError> {
    let quantity = scioxide_core::quantity!("signal_workspace_bytes");
    let values = checked_mul(values, size_of::<T>(), quantity)?;
    let real = checked_mul(real, size_of::<T::Real>(), quantity)?;
    let complex = checked_mul(complex, size_of::<Complex<T::Real>>(), quantity)?;
    checked_add(checked_add(values, real, quantity)?, complex, quantity)
}

/// Scratch storage for signal plans: one buffer each of sample, real and
/// complex elements.
///
/// One workspace serves any number of plans and calls. Its buffers grow only
/// through [`try_reserve`](Self::try_reserve) and never shrink. A call uses the
/// leading part of each buffer that its [`SignalRequirements`] name, and their
/// contents after a call are unspecified. See [Reusing plans and
/// workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_signal::{
///     IirDomain, IirFilter, IirRequest, IirState, PolyphasePlan, SignalExecutionOptions,
///     SignalLayout, SignalWorkspace, TransferFunction,
/// };
/// let fir = PolyphasePlan::fir(&[0.5_f64, 0.5])?;
/// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5], &[1.0, -0.5])?;
/// let iir = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new())?;
/// let (input, mut smoothed, mut impulse) = ([1.0, 0.0, 0.0, 0.0], [0.0; 5], [0.0; 4]);
/// let mut state = IirState::new(&iir, 1)?;
/// // Calls do not grow a workspace, so an empty one is too small.
/// let mut workspace = SignalWorkspace::new();
/// let undersized = fir.resample_into(
///     &input,
///     &mut smoothed,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// );
/// assert_eq!(undersized.unwrap_err().kind(), SciOxideErrorKind::DimensionMismatch);
/// // One reservation for the larger requirement serves both plans.
/// let layout = SignalLayout::Contiguous(input.len());
/// workspace.try_reserve(
///     &fir.workspace_requirements(layout)?
///         .max(iir.workspace_requirements(IirRequest::Filter(layout))?),
/// )?;
/// fir.resample_into(
///     &input,
///     &mut smoothed,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// iir.filter_into(
///     &input,
///     &mut impulse,
///     &mut state,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(smoothed, [0.5, 0.5, 0.0, 0.0, 0.0]);
/// assert_eq!(impulse, [0.5, 0.25, 0.125, 0.0625]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SignalWorkspace<T: NumericScalar> {
    values: Vec<T>,
    real: Vec<T::Real>,
    complex: Vec<Complex<T::Real>>,
}

impl<T: NumericScalar> Default for SignalWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed prefixes of a workspace for one operation.
pub(crate) struct SignalBuffers<'a, T: NumericScalar> {
    pub(crate) values: &'a mut [T],
    pub(crate) real: &'a mut [T::Real],
    pub(crate) complex: &'a mut [Complex<T::Real>],
}

impl<T: NumericScalar> SignalWorkspace<T> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            real: Vec::new(),
            complex: Vec::new(),
        }
    }

    /// Allocate a workspace sufficient for `requirements`.
    pub(crate) fn reserved(requirements: &SignalRequirements) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(requirements)?;
        Ok(workspace)
    }

    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        bytes::<T>(
            self.values.capacity(),
            self.real.capacity(),
            self.complex.capacity(),
        )
    }

    /// Reports whether the workspace is large enough for `requirements`, without
    /// growing it.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the capacity already held exceeds the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn check(&self, requirements: &SignalRequirements) -> Result<Capacity, SciOxideError> {
        requirements.limit(self.retained_bytes()?)?;
        Ok(
            if self.values.len() >= requirements.values
                && self.real.len() >= requirements.real
                && self.complex.len() >= requirements.complex
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the buffers to satisfy `requirements`, keeping capacity reserved
    /// for earlier plans.
    ///
    /// A failed reservation may keep the capacity it has already grown.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the resulting capacity would exceed the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    /// - `AllocationFailure` if the allocator fails.
    pub fn try_reserve(&mut self, requirements: &SignalRequirements) -> Result<(), SciOxideError> {
        requirements.limit(bytes::<T>(
            self.values.capacity().max(requirements.values),
            self.real.capacity().max(requirements.real),
            self.complex.capacity().max(requirements.complex),
        )?)?;
        grow(
            &mut self.values,
            requirements.values,
            T::ZERO,
            scioxide_core::quantity!("signal_values_workspace"),
        )?;
        grow(
            &mut self.real,
            requirements.real,
            <T::Real as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("signal_real_workspace"),
        )?;
        grow(
            &mut self.complex,
            requirements.complex,
            Complex::new(
                <T::Real as scioxide_core::AlgebraicScalar>::ZERO,
                <T::Real as scioxide_core::AlgebraicScalar>::ZERO,
            ),
            scioxide_core::quantity!("signal_complex_workspace"),
        )?;
        requirements.limit(self.retained_bytes()?)
    }

    /// Borrow exactly the declared prefixes, without growing. Byte limits are
    /// enforced by [`Self::try_reserve`] and [`Self::check`]. Execution only
    /// needs sufficient storage.
    /// # Errors
    /// Returns `DimensionMismatch` for undersized storage.
    pub(crate) fn buffers(
        &mut self,
        requirements: &SignalRequirements,
    ) -> Result<SignalBuffers<'_, T>, SciOxideError> {
        crate::util::capacity(requirements.values, self.values.len())?;
        crate::util::capacity(requirements.real, self.real.len())?;
        crate::util::capacity(requirements.complex, self.complex.len())?;
        Ok(SignalBuffers {
            values: &mut self.values[..requirements.values],
            real: &mut self.real[..requirements.real],
            complex: &mut self.complex[..requirements.complex],
        })
    }
}

fn grow<E: Clone>(
    values: &mut Vec<E>,
    len: usize,
    zero: E,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if values.len() < len {
        try_reserve(values, len - values.len(), quantity)?;
        values.resize(len, zero);
    }
    Ok(())
}

/// Execution options for signal plans whose only option is the execution policy.
///
/// The [`ExecutionPolicy`] bounds the number of workers a plan may use with the
/// executor of the call's [`ExecutionContext`](scioxide_core::ExecutionContext).
/// It defaults to [`ExecutionPolicy::Sequential`], as in every signal options
/// type. See [Execution contexts](scioxide_core#execution-contexts).
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_signal::{IirDomain, IirFilter, SignalExecutionOptions, TransferFunction};
/// assert_eq!(
///     SignalExecutionOptions::default().execution_policy(),
///     ExecutionPolicy::Sequential
/// );
/// let options = SignalExecutionOptions::new().with_execution_policy(ExecutionPolicy::Auto);
/// let transfer = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[1.0])?;
/// let plan = IirFilter::from_transfer(&transfer, options)?;
/// assert_eq!(plan.options().execution_policy(), ExecutionPolicy::Auto);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct SignalExecutionOptions {
    policy: ExecutionPolicy,
}

impl Default for SignalExecutionOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalExecutionOptions {
    /// Creates options for sequential execution.
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
        }
    }

    /// Sets the execution policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
}

/// Shape of the data in one call: a contiguous record, or every lane along one
/// axis of an N-dimensional array.
///
/// Plans take a layout in `workspace_requirements` to size the workspace for
/// that shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalLayout<'a> {
    /// One contiguous record of the given length.
    Contiguous(usize),
    /// Every lane along `axis` of an array of the given shape.
    Axis {
        /// Array shape, one extent per axis.
        shape: &'a [usize],
        /// Axis along which the plan runs.
        axis: usize,
    },
}

/// Marks whether a stream call continues the record or ends it.
///
/// See [Streams](crate#streams).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamChunk {
    /// More input follows. The call writes only the outputs that the input so far
    /// determines.
    Partial,
    /// The record ends with this chunk. The call writes the pending outputs and
    /// closes the stream until it is reset.
    Final,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn reserve_check_and_borrow_follow_the_requirements() {
        let requirements = SignalRequirements::new(3, 2, 4);
        let mut workspace = SignalWorkspace::<f64>::new();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::NeedsGrowth
        );
        assert_eq!(
            workspace.buffers(&requirements).err().unwrap().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        workspace.try_reserve(&requirements).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
        let buffers = workspace.buffers(&requirements).unwrap();
        assert_eq!(
            (
                buffers.values.len(),
                buffers.real.len(),
                buffers.complex.len()
            ),
            (3, 2, 4)
        );
        // Smaller requirements reuse retained storage; larger ones need growth.
        assert_eq!(
            workspace.check(&SignalRequirements::complex(1)).unwrap(),
            Capacity::Sufficient
        );
        assert_eq!(
            workspace.check(&SignalRequirements::new(0, 5, 0)).unwrap(),
            Capacity::NeedsGrowth
        );
    }

    #[test]
    fn byte_limits_are_resource_limits_and_combine_conservatively() {
        let requirements = SignalRequirements::values(4);
        assert_eq!(requirements.bytes::<f64>().unwrap(), 32);
        let limited = requirements.with_max_bytes(16);
        let mut workspace = SignalWorkspace::<f64>::new();
        assert_eq!(
            workspace.try_reserve(&limited).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        workspace.try_reserve(&requirements).unwrap();
        assert_eq!(
            workspace.check(&limited).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let combined = limited.max(SignalRequirements::complex(2));
        assert_eq!(
            (
                combined.values_len(),
                combined.complex_len(),
                combined.max_bytes()
            ),
            (4, 2, 16)
        );
        assert_eq!(SignalRequirements::new(0, 0, 0).bytes::<f64>().unwrap(), 0);
        assert_eq!(
            SignalRequirements::values(usize::MAX)
                .bytes::<f64>()
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    }

    #[test]
    fn complex_sample_workspaces_size_each_class() {
        let requirements = SignalRequirements::new(1, 1, 1);
        assert_eq!(requirements.bytes::<Complex<f64>>().unwrap(), 16 + 8 + 16);
        let workspace = SignalWorkspace::<Complex<f64>>::default();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::NeedsGrowth
        );
    }

    #[test]
    fn execution_options_default_to_sequential() {
        assert_eq!(
            SignalExecutionOptions::default().execution_policy(),
            ExecutionPolicy::Sequential
        );
        let parallel = ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap());
        assert_eq!(
            SignalExecutionOptions::new()
                .with_execution_policy(parallel)
                .execution_policy(),
            parallel
        );
    }
}
