//! Options and workspaces for batches of rotations and rigid motions.

use alloc::vec::Vec;
use scioxide_core::{Capacity, ExecutionPolicy, RealScalar, SciOxideError};

use crate::workspace::{add, bytes, grow, limit, require};
use crate::{EulerAngles, Rotation};

/// Options for batch transforms: the execution policy and a workspace byte
/// limit.
///
/// They apply to [`Rotation::apply_batch_into`](crate::Rotation::apply_batch_into),
/// [`RigidTransform::apply_batch_into`](crate::RigidTransform::apply_batch_into),
/// [`Rotation::as_euler_batch_into`](crate::Rotation::as_euler_batch_into) and the
/// `evaluate_rotations_into` methods of [`Slerp`](crate::Slerp) and
/// [`RotationSpline`](crate::RotationSpline).
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_spatial::TransformOptions;
/// let options = TransformOptions::new()
///     .with_execution_policy(ExecutionPolicy::Auto)
///     .with_max_workspace_bytes(4096);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Auto);
/// assert_eq!(options.max_workspace_bytes(), 4096);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TransformOptions {
    policy: ExecutionPolicy,
    max_workspace_bytes: usize,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformOptions {
    /// Returns sequential execution without a byte limit.
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
            max_workspace_bytes: usize::MAX,
        }
    }
    /// Sets the execution policy.
    ///
    /// With `Auto`, a batch of at least 256 outputs runs in parallel when each
    /// worker gets at least the executor's minimum work. The work is estimated
    /// as 18 units per transformed point, 64 per Euler extraction and 128 per
    /// interpolated rotation.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the limit on the bytes a [`TransformWorkspace`] may hold, including
    /// capacity from earlier, larger reservations.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the workspace byte limit. The fixed-size part of the workspace
    /// is not counted.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
}

/// The kind and size of a transform batch, used to size a
/// [`TransformWorkspace`].
///
/// ```
/// use scioxide_spatial::{TransformOptions, TransformWorkspace, TransformWorkspaceRequest as Request};
/// let options = TransformOptions::new();
/// for request in [
///     Request::apply(4, options),
///     Request::euler(4, options),
///     Request::rotations(4, options),
/// ] {
///     let requirements = TransformWorkspace::<f64>::workspace_requirements(request)?;
///     assert!(requirements.bytes::<f64>()? > 0);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TransformWorkspaceRequest {
    lengths: [usize; 3],
    options: TransformOptions,
}

impl TransformWorkspaceRequest {
    /// Requests space to apply a rotation or rigid motion to `count` points.
    pub const fn apply(count: usize, options: TransformOptions) -> Self {
        Self {
            lengths: [count, 0, 0],
            options,
        }
    }
    /// Requests space to compute Euler angles of `count` rotations.
    pub const fn euler(count: usize, options: TransformOptions) -> Self {
        Self {
            lengths: [0, count, 0],
            options,
        }
    }
    /// Requests space to evaluate a slerp or rotation spline at `count` times.
    pub const fn rotations(count: usize, options: TransformOptions) -> Self {
        Self {
            lengths: [0, 0, count],
            options,
        }
    }
    fn requirements<S: RealScalar>(self) -> Result<TransformWorkspaceRequirements, SciOxideError> {
        let requirements = TransformWorkspaceRequirements {
            lengths: self.lengths,
            max_workspace_bytes: self.options.max_workspace_bytes,
        };
        requirements.bytes::<S>()?;
        Ok(requirements)
    }
}

/// Workspace sizes for one transform batch, and the byte limit.
#[derive(Debug, Clone, Copy)]
pub struct TransformWorkspaceRequirements {
    lengths: [usize; 3],
    max_workspace_bytes: usize,
}

fn storage_bytes<S: RealScalar>(
    [points, euler, rotations]: [usize; 3],
) -> Result<usize, SciOxideError> {
    add(
        add(bytes::<[S; 3]>(points)?, bytes::<EulerAngles<S>>(euler)?)?,
        bytes::<Rotation<S>>(rotations)?,
    )
}

impl TransformWorkspaceRequirements {
    /// Returns the bytes needed for scalar type `S`, without the fixed-size
    /// part of the workspace.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size exceeds `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit.
    pub fn bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let bytes = storage_bytes::<S>(self.lengths)?;
        limit(bytes, self.max_workspace_bytes)?;
        Ok(bytes)
    }
}

/// Workspace for transform batches.
///
/// A batch writes its results here first and copies them to the output only
/// after every result is computed, so the output is unchanged on error.
/// Batches never grow the workspace. All capacity it holds counts toward the
/// byte limit, including buffers that a request does not use.
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_spatial::{TransformOptions, TransformWorkspace, TransformWorkspaceRequest};
/// let request = TransformWorkspaceRequest::apply(8, TransformOptions::new());
/// let requirements = TransformWorkspace::<f64>::workspace_requirements(request)?;
/// let mut workspace = TransformWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Default)]
pub struct TransformWorkspace<S: RealScalar> {
    pub(crate) points: Vec<[S; 3]>,
    pub(crate) euler: Vec<EulerAngles<S>>,
    pub(crate) rotations: Vec<Rotation<S>>,
}

impl<S: RealScalar> TransformWorkspace<S> {
    /// Returns the workspace sizes for a batch.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size exceeds `usize`.
    /// - `ResourceLimit` if the workspace would exceed the byte limit.
    pub fn workspace_requirements(
        request: TransformWorkspaceRequest,
    ) -> Result<TransformWorkspaceRequirements, SciOxideError> {
        request.requirements::<S>()
    }

    /// Allocates a workspace for a batch.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`], and `AllocationFailure` if the
    /// storage cannot be reserved.
    ///
    /// ```
    /// use scioxide_spatial::{TransformOptions, TransformWorkspace, TransformWorkspaceRequest};
    /// let request = TransformWorkspaceRequest::rotations(4, TransformOptions::new());
    /// let workspace = TransformWorkspace::<f64>::workspace(request)?;
    /// assert_eq!(
    ///     workspace.check(&TransformWorkspace::<f64>::workspace_requirements(request)?)?,
    ///     scioxide_core::Capacity::Sufficient
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(request: TransformWorkspaceRequest) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            points: Vec::new(),
            euler: Vec::new(),
            rotations: Vec::new(),
        }
    }
    fn capacities(&self) -> [usize; 3] {
        [
            self.points.capacity(),
            self.euler.capacity(),
            self.rotations.capacity(),
        ]
    }
    /// Reports whether the workspace meets `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size exceeds `usize`.
    /// - `ResourceLimit` if the capacity already held exceeds the byte limit.
    #[inline]
    pub fn check(
        &self,
        requirements: &TransformWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        requirements.bytes::<S>()?;
        limit(
            storage_bytes::<S>(self.capacities())?,
            requirements.max_workspace_bytes,
        )?;
        Ok(
            if [self.points.len(), self.euler.len(), self.rotations.len()]
                .into_iter()
                .zip(requirements.lengths)
                .all(|(actual, required)| actual >= required)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to meet `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved. The workspace may then keep part of the new capacity.
    pub fn try_reserve(
        &mut self,
        requirements: &TransformWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        let capacities = self.capacities();
        let projected = core::array::from_fn(|i| capacities[i].max(requirements.lengths[i]));
        limit(
            storage_bytes::<S>(projected)?,
            requirements.max_workspace_bytes,
        )?;
        let [points, euler, rotations] = requirements.lengths;
        grow(&mut self.points, points, [S::ZERO; 3])?;
        grow(
            &mut self.euler,
            euler,
            EulerAngles {
                angles: [S::ZERO; 3],
                gimbal_lock: false,
            },
        )?;
        grow(&mut self.rotations, rotations, Rotation::identity())?;
        let _ = self.check(requirements)?;
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn require(&self, request: TransformWorkspaceRequest) -> Result<(), SciOxideError> {
        // check validates bytes and retained capacity together, once per call.
        let requirements = TransformWorkspaceRequirements {
            lengths: request.lengths,
            max_workspace_bytes: request.options.max_workspace_bytes,
        };
        if self.check(&requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        for (actual, required) in [self.points.len(), self.euler.len(), self.rotations.len()]
            .into_iter()
            .zip(requirements.lengths)
        {
            require(actual, required)?;
        }
        Ok(())
    }
}
