//! Options and workspaces for distance batches.

use alloc::vec::Vec;
use scioxide_core::{Capacity, ExecutionPolicy, RealScalar, SciOxideError};

pub(crate) fn add(a: usize, b: usize) -> Result<usize, SciOxideError> {
    a.checked_add(b).ok_or_else(overflow)
}

pub(crate) fn bytes<T>(len: usize) -> Result<usize, SciOxideError> {
    len.checked_mul(core::mem::size_of::<T>())
        .ok_or_else(overflow)
}

pub(crate) fn overflow() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!("spatial_workspace"))
}

pub(crate) fn limit(required: usize, maximum: usize) -> Result<(), SciOxideError> {
    if required > maximum {
        Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("workspace_bytes"),
            required,
            maximum,
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn grow<T: Clone>(
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), SciOxideError> {
    bytes::<T>(len)?;
    values
        .try_reserve_exact(len.saturating_sub(values.len()))
        .map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("spatial_workspace"), len)
        })?;
    if len > values.len() {
        values.resize(len, value);
    }
    Ok(())
}

pub(crate) fn require(actual: usize, required: usize) -> Result<(), SciOxideError> {
    if actual < required {
        Err(SciOxideError::dimension_mismatch([required], [actual]))
    } else {
        Ok(())
    }
}

/// Options for distance batches and matrix conversions: the execution policy
/// and a workspace byte limit.
#[derive(Debug, Clone, Copy)]
pub struct DistanceOptions {
    policy: ExecutionPolicy,
    max_workspace_bytes: usize,
}

impl Default for DistanceOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceOptions {
    /// Returns sequential execution without a byte limit.
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
            max_workspace_bytes: usize::MAX,
        }
    }
    /// Sets the execution policy.
    ///
    /// With `Auto`, a batch of at least 256 output values runs in parallel when
    /// each worker gets at least the executor's minimum work. The work is
    /// estimated as `d` units per pair of `d`-dimensional points (`d²` for
    /// Mahalanobis, `16 d` for Jensen–Shannon) and one per converted matrix
    /// entry.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the limit on the bytes a [`DistanceWorkspace`] may hold, including
    /// capacity from earlier, larger reservations.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the workspace byte limit.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
}

/// The kind and size of a distance batch, used to size a [`DistanceWorkspace`].
#[derive(Debug, Clone, Copy)]
pub struct DistanceWorkspaceRequest {
    shape: DistanceShape,
    options: DistanceOptions,
}

#[derive(Debug, Clone, Copy)]
enum DistanceShape {
    Pairwise(usize, usize),
    Condensed(usize),
    Square(usize),
}

impl DistanceWorkspaceRequest {
    /// Requests space for [`PreparedDistance::pairwise_into`](crate::PreparedDistance::pairwise_into) with `left`
    /// and `right` points.
    pub const fn pairwise(left: usize, right: usize, options: DistanceOptions) -> Self {
        Self {
            shape: DistanceShape::Pairwise(left, right),
            options,
        }
    }
    /// Requests space for [`PreparedDistance::condensed_into`](crate::PreparedDistance::condensed_into) or
    /// [`square_to_condensed_into`](crate::square_to_condensed_into) with
    /// `points` points.
    pub const fn condensed(points: usize, options: DistanceOptions) -> Self {
        Self {
            shape: DistanceShape::Condensed(points),
            options,
        }
    }
    /// Requests space for
    /// [`condensed_to_square_into`](crate::condensed_to_square_into) with
    /// `points` points.
    pub const fn square(points: usize, options: DistanceOptions) -> Self {
        Self {
            shape: DistanceShape::Square(points),
            options,
        }
    }
    fn requirements(self) -> Result<DistanceWorkspaceRequirements, SciOxideError> {
        let len = match self.shape {
            DistanceShape::Pairwise(m, n) => m.checked_mul(n).ok_or_else(overflow)?,
            DistanceShape::Condensed(n) => crate::condensed_len(n)?,
            DistanceShape::Square(n) => n.checked_mul(n).ok_or_else(overflow)?,
        };
        Ok(DistanceWorkspaceRequirements {
            len,
            max_workspace_bytes: self.options.max_workspace_bytes,
        })
    }
}

/// Workspace size for one distance batch, and the byte limit.
#[derive(Debug, Clone, Copy)]
pub struct DistanceWorkspaceRequirements {
    len: usize,
    max_workspace_bytes: usize,
}
impl DistanceWorkspaceRequirements {
    /// Returns the number of scalar values the workspace must hold.
    pub const fn scalar_len(self) -> usize {
        self.len
    }
    /// Returns the bytes needed for scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size exceeds `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit.
    pub fn bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let bytes = bytes::<S>(self.len)?;
        limit(bytes, self.max_workspace_bytes)?;
        Ok(bytes)
    }
}

/// Workspace for distance batches and matrix conversions.
///
/// A batch writes its results here first and copies them to the output only
/// after every result is computed, so the output is unchanged on error.
/// Batches never grow the workspace.
#[derive(Debug, Default)]
pub struct DistanceWorkspace<S: RealScalar> {
    pub(crate) values: Vec<S>,
}
impl<S: RealScalar> DistanceWorkspace<S> {
    /// Returns the workspace size for a batch.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the number of results or bytes exceeds `usize`.
    /// - `ResourceLimit` if the workspace would exceed the byte limit.
    pub fn workspace_requirements(
        request: DistanceWorkspaceRequest,
    ) -> Result<DistanceWorkspaceRequirements, SciOxideError> {
        let requirements = request.requirements()?;
        requirements.bytes::<S>()?;
        Ok(requirements)
    }

    /// Allocates a workspace for a batch.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`], and `AllocationFailure` if the
    /// storage cannot be reserved.
    ///
    /// ```
    /// use scioxide_spatial::{DistanceOptions, DistanceWorkspace, DistanceWorkspaceRequest};
    /// let request = DistanceWorkspaceRequest::square(3, DistanceOptions::new());
    /// let requirements = DistanceWorkspace::<f64>::workspace_requirements(request)?;
    /// let workspace = DistanceWorkspace::<f64>::workspace(request)?;
    /// assert_eq!(requirements.scalar_len(), 9);
    /// assert_eq!(
    ///     workspace.check(&requirements)?,
    ///     scioxide_core::Capacity::Sufficient
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(request: DistanceWorkspaceRequest) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self { values: Vec::new() }
    }
    /// Reports whether the workspace meets `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size exceeds `usize`.
    /// - `ResourceLimit` if the capacity already held exceeds the byte limit.
    pub fn check(
        &self,
        requirements: &DistanceWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        requirements.bytes::<S>()?;
        limit(
            bytes::<S>(self.values.capacity())?,
            requirements.max_workspace_bytes,
        )?;
        Ok(if self.values.len() >= requirements.len {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to meet `requirements`. Capacity from earlier
    /// reservations is kept.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved. The workspace may then keep part of the new capacity.
    pub fn try_reserve(
        &mut self,
        requirements: &DistanceWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        limit(
            bytes::<S>(self.values.capacity().max(requirements.len))?,
            requirements.max_workspace_bytes,
        )?;
        grow(&mut self.values, requirements.len, S::ZERO)?;
        let _ = self.check(requirements)?;
        Ok(())
    }
    pub(crate) fn prepare(
        &mut self,
        request: DistanceWorkspaceRequest,
    ) -> Result<&mut [S], SciOxideError> {
        let requirements = request.requirements()?;
        let _ = self.check(&requirements)?;
        require(self.values.len(), requirements.len)?;
        Ok(&mut self.values[..requirements.len])
    }
}
