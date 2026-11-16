//! Workspaces and workspace requirements for prepared transforms.
use crate::validation::sufficient;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut};
use scioxide_core::backend::checked_mul;
use scioxide_core::{Capacity, Complex, RealScalar, SciOxideError, SolveState};

/// Selects the workspace size for a single transform or for a transform along
/// one axis of an N-D array.
///
/// The chirp-z, nonuniform, discrete Hankel and FFTLog plans take this request
/// in `workspace_requirements` and `workspace`. Use `()` or [`Self::Plain`] for
/// `execute_into`, and [`Self::Axis`] for `execute_axis_into`. An axis
/// workspace also holds the whole output, because all lanes are written to the
/// output together after every lane has succeeded.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_fft::{AxisWorkspaceRequest, CztContour, CztPlan, CztPlanSpec, Normalization};
/// // Three-point DFTs down the columns of a 3 × 2 array of ones; each DC bin is 3.
/// let contour = CztContour::<f64>::dft(3)?;
/// let plan = CztPlan::new(CztPlanSpec::new(3, 3, contour, Normalization::Unscaled))?;
/// let mut work = plan.workspace(AxisWorkspaceRequest::Axis {
///     shape: &[3, 2],
///     axis: 0,
/// })?;
/// let layout = StridedLayout::new(&[3, 2], &[2, 1])?;
/// let mut output = [Complex::new(0.0, 0.0); 6];
/// let x = NdView::new(&[1.0; 6], layout)?;
/// let y = NdViewMut::new(&mut output, layout)?;
/// plan.execute_axis_into(x, 0, y, &mut work, ExecutionContext::sequential())?;
/// assert!((output[0] - Complex::new(3.0, 0.0)).norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub enum AxisWorkspaceRequest<'a> {
    /// A single transform of the plan's length.
    Plain,
    /// A transform along `axis` of an array with shape `shape`.
    Axis {
        /// Shape of the whole input array, including the batch axes.
        shape: &'a [usize],
        /// Axis to transform.
        axis: usize,
    },
}
impl From<()> for AxisWorkspaceRequest<'_> {
    fn from((): ()) -> Self {
        Self::Plain
    }
}

/// Holds reusable complex scratch storage for prepared transforms.
///
/// Get one from a plan's `workspace` method, or create one with [`Self::new`]
/// and size it with [`Self::try_reserve`]. One workspace can serve several
/// plans: it grows to the largest requirement and never shrinks. Its full
/// capacity, including storage reserved for earlier plans, counts toward a
/// plan's [byte limit](FftWorkspaceRequirements::byte_limit).
///
/// The workspace dereferences to a slice of complex values, so it can be passed
/// to methods that take a scratch slice. Methods that count work toward a
/// [solve scope](scioxide_core#solve-accounting) take the workspace itself,
/// because it also holds that solve state. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
#[derive(Debug, Default)]
pub struct FftWorkspace<S: RealScalar> {
    pub(crate) values: Vec<Complex<S>>,
    pub(crate) state: SolveState,
}

impl<S: RealScalar> FftWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            state: SolveState::new(),
        }
    }

    /// Creates a workspace that uses `values` as its storage.
    ///
    /// The capacity of `values` is checked against a plan's byte limit when the
    /// workspace is checked, reserved or used.
    pub fn from_vec(values: Vec<Complex<S>>) -> Self {
        Self {
            values,
            state: SolveState::new(),
        }
    }

    /// Reports whether the workspace meets `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the current capacity exceeds the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if the capacity in bytes overflows `usize`.
    pub fn check(
        &self,
        requirements: &FftWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        requirements.check_capacity::<S>(self.values.capacity())?;
        Ok(if self.values.len() >= requirements.complex_len() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows the workspace to meet `requirements`, keeping its existing
    /// capacity.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the resulting capacity would exceed the byte limit
    ///   of `requirements`.
    /// - `AllocationFailure` if the storage cannot grow.
    /// - `SizeOverflow` if the capacity in bytes overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &FftWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        requirements.check_capacity::<S>(self.values.capacity().max(requirements.complex_len()))?;
        let additional = requirements.complex_len().saturating_sub(self.values.len());
        self.values.try_reserve_exact(additional).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("complex_workspace"),
                requirements.complex_len(),
            )
        })?;
        requirements.check_capacity::<S>(self.values.capacity())?;
        if self.values.len() < requirements.complex_len() {
            self.values
                .resize(requirements.complex_len(), Complex::new(S::ZERO, S::ZERO));
        }
        Ok(())
    }

    /// Validate storage and run `operation` in a solve epoch owned by this
    /// workspace.
    pub(crate) fn run<R>(
        &mut self,
        requirements: &FftWorkspaceRequirements,
        length: usize,
        budgets: scioxide_core::SolveBudgets,
        scope: scioxide_core::SolveScope<'_, '_>,
        operation: impl FnOnce(
            &mut [Complex<S>],
            &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<R, SciOxideError>,
    ) -> Result<R, SciOxideError> {
        sufficient(self.values.len(), requirements.complex_len())?;
        let _ = self.check(requirements)?;
        scope.execution().check()?;
        scope.run(&mut self.state, identity(length), budgets, |child| {
            operation(&mut self.values, child)
        })
    }
}
impl<S: RealScalar> Deref for FftWorkspace<S> {
    type Target = [Complex<S>];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl<S: RealScalar> DerefMut for FftWorkspace<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

pub(crate) fn allocate<S: RealScalar>(
    requirements: &FftWorkspaceRequirements,
) -> Result<FftWorkspace<S>, SciOxideError> {
    let mut workspace = FftWorkspace::new();
    workspace.try_reserve(requirements)?;
    Ok(workspace)
}

/// Compose workspace byte limits for `total` bytes embedding child regions
/// given as `(byte_limit, bytes)`: each child keeps its own limit, bytes
/// outside it are accepted, and `configured` caps the whole. Unbounded
/// children stay unbounded.
pub(crate) fn nested_byte_limit(
    configured: usize,
    total: usize,
    children: impl IntoIterator<Item = (usize, usize)>,
) -> usize {
    children
        .into_iter()
        .fold(configured, |limit, (child_limit, child)| {
            limit.min(child_limit.saturating_add(total.saturating_sub(child)))
        })
}

pub(crate) fn identity(length: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("fft", "transform", 1),
        length as u128,
        0,
    )
}

/// Holds the real and complex scratch buffers of a
/// [`RealTransformNdPlan`](crate::RealTransformNdPlan).
///
/// It follows the same rules as [`FftWorkspace`]: buffers grow and never
/// shrink, and their combined capacity counts toward the plan's byte limit.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_fft::{
///     DctType, FftDirection, Normalization, RealTransformKind, RealTransformNdPlan,
///     RealTransformNdPlanSpec, RealTransformNdWorkspace,
/// };
/// let (forward, ortho) = (FftDirection::Forward, Normalization::Orthonormal);
/// let types = [RealTransformKind::Dct(DctType::II); 2];
/// let spec = RealTransformNdPlanSpec::new(&[2, 2], &[0, 1], &types, forward, ortho)?;
/// let plan = RealTransformNdPlan::<f64>::new(spec)?;
/// let requirements = plan.workspace_requirements(())?;
/// let mut workspace = RealTransformNdWorkspace::<f64>::new();
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Default)]
pub struct RealTransformNdWorkspace<S: RealScalar> {
    pub(crate) real: Vec<S>,
    pub(crate) complex: Vec<Complex<S>>,
}
impl<S: RealScalar> RealTransformNdWorkspace<S> {
    /// Creates empty buffers without allocating.
    pub const fn new() -> Self {
        Self {
            real: Vec::new(),
            complex: Vec::new(),
        }
    }
    /// Reports whether both buffers meet `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the combined capacity exceeds the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if the capacity in bytes overflows `usize`.
    pub fn check(
        &self,
        requirements: &crate::RealTransformNdWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        requirements.check_capacity::<S>(self.real.capacity(), self.complex.capacity())?;
        Ok(
            if self.real.len() >= requirements.real_len()
                && self.complex.len() >= requirements.complex().complex_len()
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows both buffers to meet `requirements`, keeping their existing
    /// capacity.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the combined capacity would exceed the byte limit
    ///   of `requirements`.
    /// - `AllocationFailure` if either buffer cannot grow.
    /// - `SizeOverflow` if the capacity in bytes overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &crate::RealTransformNdWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        requirements.check_capacity::<S>(
            self.real.capacity().max(requirements.real_len()),
            self.complex
                .capacity()
                .max(requirements.complex().complex_len()),
        )?;
        self.real
            .try_reserve_exact(requirements.real_len().saturating_sub(self.real.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("real_workspace"),
                    requirements.real_len(),
                )
            })?;
        self.complex
            .try_reserve_exact(
                requirements
                    .complex()
                    .complex_len()
                    .saturating_sub(self.complex.len()),
            )
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("complex_workspace"),
                    requirements.complex().complex_len(),
                )
            })?;
        requirements.check_capacity::<S>(self.real.capacity(), self.complex.capacity())?;
        if self.real.len() < requirements.real_len() {
            self.real.resize(requirements.real_len(), S::ZERO);
        }
        if self.complex.len() < requirements.complex().complex_len() {
            self.complex.resize(
                requirements.complex().complex_len(),
                Complex::new(S::ZERO, S::ZERO),
            );
        }
        Ok(())
    }
}

/// Describes the complex scratch that a prepared plan needs.
///
/// Plans return it from `workspace_requirements`. Pass it to
/// [`FftWorkspace::check`] or [`FftWorkspace::try_reserve`], or allocate
/// [`Self::complex_len`] values yourself for the slice-based `execute_into`
/// methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct FftWorkspaceRequirements {
    complex_len: usize,
    bytes: usize,
    alignment: usize,
    limit: usize,
}

impl FftWorkspaceRequirements {
    /// Describe a contiguous complex buffer using the scalar's size and alignment.
    /// # Errors
    /// Returns `SizeOverflow` when the byte count cannot be represented.
    pub(crate) fn new<S: scioxide_core::RealScalar>(
        complex_len: usize,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            complex_len,
            bytes: checked_mul(
                complex_len,
                size_of::<Complex<S>>(),
                scioxide_core::quantity!("fft_scratch_byte_requirement"),
            )?,
            alignment: align_of::<Complex<S>>(),
            limit: usize::MAX,
        })
    }

    pub(crate) fn with_limit(mut self, limit: usize) -> Result<Self, SciOxideError> {
        self.limit = limit;
        if self.bytes > limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                self.bytes,
                limit,
            ));
        }
        Ok(self)
    }
    pub(crate) fn check_capacity<S: scioxide_core::RealScalar>(
        &self,
        capacity: usize,
    ) -> Result<(), SciOxideError> {
        let bytes = checked_mul(
            capacity,
            size_of::<Complex<S>>(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if bytes > self.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                self.limit,
            ));
        }
        Ok(())
    }
    /// Returns the largest workspace capacity, in bytes, that the plan
    /// accepts; `usize::MAX` if the plan has no limit.
    pub const fn byte_limit(self) -> usize {
        self.limit
    }
    /// Returns the required number of complex values.
    pub const fn complex_len(self) -> usize {
        self.complex_len
    }

    /// Returns the required scratch size in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }

    /// Returns the required alignment in bytes.
    pub const fn alignment(self) -> usize {
        self.alignment
    }
}

/// Describes the real and complex scratch that a
/// [`RealTransformNdPlan`](crate::RealTransformNdPlan) needs.
///
/// Pass it to [`RealTransformNdWorkspace::try_reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealTransformNdWorkspaceRequirements {
    real_len: usize,
    real_bytes: usize,
    real_alignment: usize,
    complex: crate::FftWorkspaceRequirements,
    bytes: usize,
    limit: usize,
}
impl RealTransformNdWorkspaceRequirements {
    /// Size both buffers, reporting byte overflow.
    pub(crate) fn new<S: RealScalar>(
        real_len: usize,
        complex_len: usize,
    ) -> Result<Self, SciOxideError> {
        let complex = crate::FftWorkspaceRequirements::new::<S>(complex_len)?;
        let real_bytes = scioxide_core::backend::checked_mul(
            real_len,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("real_transform_scratch_bytes"),
        )?;
        let bytes = scioxide_core::backend::checked_add(
            real_bytes,
            complex.bytes(),
            scioxide_core::quantity!("real_transform_scratch_bytes"),
        )?;
        Ok(Self {
            real_len,
            real_bytes,
            real_alignment: core::mem::align_of::<S>(),
            complex,
            bytes,
            limit: usize::MAX,
        })
    }
    pub(crate) fn with_limit(mut self, limit: usize) -> Result<Self, SciOxideError> {
        self.limit = limit;
        self.check_bytes(self.bytes)?;
        Ok(self)
    }
    fn check_bytes(&self, bytes: usize) -> Result<(), SciOxideError> {
        if bytes > self.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                self.limit,
            ));
        }
        Ok(())
    }
    pub(crate) fn check_capacity<S: RealScalar>(
        &self,
        real: usize,
        complex: usize,
    ) -> Result<(), SciOxideError> {
        let resource = scioxide_core::quantity!("workspace_bytes");
        let real = scioxide_core::backend::checked_mul(real, core::mem::size_of::<S>(), resource)?;
        let complex = scioxide_core::backend::checked_mul(
            complex,
            core::mem::size_of::<Complex<S>>(),
            resource,
        )?;
        self.check_bytes(scioxide_core::backend::checked_add(
            real, complex, resource,
        )?)
    }
    /// Returns the number of real elements.
    pub const fn real_len(self) -> usize {
        self.real_len
    }
    /// Returns the size of the real buffer in bytes.
    pub const fn real_bytes(self) -> usize {
        self.real_bytes
    }
    /// Returns the alignment of the real buffer in bytes.
    pub const fn real_alignment(self) -> usize {
        self.real_alignment
    }
    /// Returns the requirements of the complex buffer.
    pub const fn complex(self) -> crate::FftWorkspaceRequirements {
        self.complex
    }
    /// Returns the largest combined capacity of both buffers, in bytes, that
    /// the plan accepts; `usize::MAX` if the plan has no limit.
    pub const fn byte_limit(self) -> usize {
        self.limit
    }
    /// Returns the required size of both buffers together, in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_requirement_arithmetic_and_reusable_epochs() {
        assert_eq!(
            FftWorkspaceRequirements::new::<f64>(usize::MAX)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::SizeOverflow
        );
        let requirements = FftWorkspaceRequirements::new::<f64>(1)
            .unwrap()
            .with_limit(size_of::<Complex<f64>>())
            .unwrap();
        assert_eq!(requirements.byte_limit(), requirements.bytes());
        let mut workspace = allocate::<f64>(&requirements).unwrap();
        for epoch in 1..=2 {
            workspace
                .run(
                    &requirements,
                    1,
                    scioxide_core::SolveBudgets::unbounded(),
                    scioxide_core::ExecutionContext::sequential().into(),
                    |_, child| child.record_iteration()?.require(),
                )
                .unwrap();
            assert_eq!(workspace.state.epoch().unwrap().get(), epoch);
            assert_eq!(workspace.state.counts().own().iterations(), 1);
        }
    }

    #[test]
    fn nested_limits_compose_children_under_the_configured_cap() {
        assert_eq!(nested_byte_limit(100, 64, []), 100);
        assert_eq!(nested_byte_limit(usize::MAX, 64, [(16, 16)]), 64);
        assert_eq!(nested_byte_limit(40, 64, [(16, 16)]), 40);
        assert_eq!(nested_byte_limit(usize::MAX, 64, [(32, 16), (8, 32)]), 40);
        assert_eq!(
            nested_byte_limit(usize::MAX, 64, [(usize::MAX, 16)]),
            usize::MAX
        );
        assert_eq!(nested_byte_limit(usize::MAX, 0, [(24, 16)]), 24);
    }
}
