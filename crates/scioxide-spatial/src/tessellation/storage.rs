//! Storage accounting for tessellation construction buffers.
use crate::tessellation::{Result, limit};
use alloc::vec::Vec;
use core::cell::Cell;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{Quantity, SciOxideError};

const BYTES: Quantity = scioxide_core::quantity!("geometry_storage_bytes");

/// Peak storage used by one tessellation construction.
///
/// The count covers buffer capacity and scratch, including temporary storage
/// that is freed before construction ends. It excludes the input points and
/// allocator overhead. It describes this construction only; another input
/// can need more. In tests on random and degenerate inputs the recorded value
/// is at most twice the peak measured by the allocator.
///
/// ```
/// use scioxide_spatial::{ConvexHull, GeometryLimits, GeometryOptions, PointSet};
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let points = PointSet::from_rows(&sites, 2)?;
/// let bytes = ConvexHull::new(&points)?
///     .construction_requirements()
///     .bytes();
/// // The recorded peak is sufficient as an explicit storage limit for this input.
/// let exact = GeometryOptions::new().with_limits(GeometryLimits::new().with_max_storage_bytes(bytes));
/// assert_eq!(ConvexHull::with_options(&points, exact)?.facet_count(), 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryConstructionRequirements {
    bytes: usize,
}
impl GeometryConstructionRequirements {
    /// Returns the peak storage in bytes, including temporary storage.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Debug)]
pub(super) struct Storage {
    live: Cell<usize>,
    peak: Cell<usize>,
    maximum: usize,
}
impl Storage {
    pub(super) fn new(maximum: usize, retained: usize) -> Result<Self> {
        limit(BYTES, retained, maximum)?;
        Ok(Self {
            live: Cell::new(retained),
            peak: Cell::new(retained),
            maximum,
        })
    }
    pub(super) fn charge(&self, bytes: usize) -> Result<()> {
        let live = checked_add(self.live.get(), bytes, BYTES)?;
        limit(BYTES, live, self.maximum)?;
        self.live.set(live);
        self.peak.set(self.peak.get().max(live));
        Ok(())
    }
    pub(super) fn release(&self, bytes: usize) {
        self.live.set(self.live.get() - bytes);
    }
    pub(super) fn requirements(&self) -> GeometryConstructionRequirements {
        GeometryConstructionRequirements {
            bytes: self.peak.get(),
        }
    }
    pub(super) fn frame(&self) -> Frame<'_> {
        Frame {
            storage: self,
            baseline: self.live.get(),
        }
    }
    pub(super) fn capacity<T>(&self, capacity: usize, quantity: Quantity) -> Result<Vec<T>> {
        let bytes = checked_mul(capacity, size_of::<T>(), quantity)?;
        self.charge(bytes)?;
        let result = scioxide_core::backend::try_with_capacity(capacity, quantity)?;
        self.charge(retained(&result) - bytes)?;
        Ok(result)
    }
    pub(super) fn filled<T: Clone>(
        &self,
        len: usize,
        value: T,
        quantity: Quantity,
    ) -> Result<Vec<T>> {
        let mut values = self.capacity(len, quantity)?;
        values.resize(len, value);
        Ok(values)
    }
    pub(super) fn copy<T: Clone>(&self, input: &[T], quantity: Quantity) -> Result<Vec<T>> {
        let mut values = self.capacity(input.len(), quantity)?;
        values.extend_from_slice(input);
        Ok(values)
    }
    pub(super) fn reserve<T>(&self, values: &mut Vec<T>, additional: usize) -> Result<()> {
        let required = checked_add(values.len(), additional, BYTES)?;
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = required
            .max(checked_mul(values.capacity(), 2, BYTES)?)
            .max(4);
        let before = retained(values);
        let bytes = checked_mul(capacity, size_of::<T>(), BYTES)?;
        self.charge(bytes - before)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| SciOxideError::allocation_failure(BYTES, capacity))?;
        self.charge(retained(values) - bytes)?;
        Ok(())
    }
    pub(super) fn linalg(
        &self,
        dimension: usize,
    ) -> Result<scioxide_linalg::LinalgWorkspace<'static, f64>> {
        use scioxide_linalg::{
            LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, LuWorkspaceRequest,
        };
        let options = LinalgExecutionOptions::new();
        let factor = LuDecomposition::<f64>::workspace_requirements(&LuWorkspaceRequest::factor(
            dimension, &options,
        ))?;
        let solve = LuDecomposition::<f64>::workspace_requirements(
            &LuWorkspaceRequest::solve_matrix(dimension, dimension, &options),
        )?;
        let admitted = checked_add(
            factor.required_bytes::<f64>()?,
            solve.required_bytes::<f64>()?,
            BYTES,
        )?;
        self.charge(admitted)?;
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&factor)?;
        workspace.try_reserve(&solve)?;
        let actual = workspace.retained_bytes()?;
        if actual <= admitted {
            self.release(admitted - actual);
        } else {
            self.charge(actual - admitted)?;
        }
        // The factor output grows on the first factorization and is reused.
        self.charge(LuDecomposition::<f64>::factor_storage_bytes(dimension)?)?;
        Ok(workspace)
    }
}

/// A phase releases its temporary charges on every exit. Successful phases
/// explicitly transfer the output they keep and release consumed input.
pub(super) struct Frame<'a> {
    storage: &'a Storage,
    baseline: usize,
}
impl Frame<'_> {
    pub(super) fn keep(mut self, removed: usize, retained: usize) {
        self.baseline = self.baseline - removed + retained;
    }
}
impl Drop for Frame<'_> {
    fn drop(&mut self) {
        self.storage.live.set(self.baseline);
    }
}

pub(super) fn retained<T>(values: &Vec<T>) -> usize {
    // Vec capacity is already constrained by the address space and element size.
    values.capacity() * size_of::<T>()
}
