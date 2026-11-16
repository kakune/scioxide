//! All-or-nothing updates of the values of a sparse matrix.
use alloc::vec::Vec;
use scioxide_core::{Capacity, NonFiniteOrigin, NumericScalar, SciOxideError};

/// Holds the storage that `update_values` needs for one matrix: one scalar
/// per stored entry.
///
/// Returned by the `workspace_requirements(())` method of each matrix type.
/// See [`SparseValueWorkspace`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseValueRequirements {
    len: usize,
    bytes: usize,
}
impl SparseValueRequirements {
    pub(crate) fn new<S>(len: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            len,
            bytes: scioxide_core::backend::checked_mul(
                len,
                core::mem::size_of::<S>(),
                scioxide_core::quantity!("sparse_value_workspace"),
            )?,
        })
    }
    /// Returns the number of scalars, equal to the number of stored values.
    pub const fn scalar_len(self) -> usize {
        self.len
    }
    /// Returns the number of bytes for those scalars, without allocator
    /// overhead.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
}

/// Holds the new values while `update_values` fills and checks them.
///
/// Create one with a matrix's `workspace(())`, or reserve an existing one with
/// [`try_reserve`](Self::try_reserve). The new values are copied into the
/// matrix only after all of them are checked, so the workspace keeps its
/// storage and serves any matrix with at most as many stored values.
///
/// ```
/// use scioxide_core::{Capacity, SciOxideError, SciOxideErrorKind};
/// use scioxide_sparse::{CsrMatrix, SparseValueWorkspace};
///
/// let mut a = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0])?;
/// let needed = a.workspace_requirements(())?;
/// assert_eq!(needed.scalar_len(), 2);
/// let mut ws = SparseValueWorkspace::new();
/// assert_eq!(ws.check(&needed)?, Capacity::NeedsGrowth);
/// ws.try_reserve(&needed)?;
/// assert_eq!(ws.check(&needed)?, Capacity::Sufficient);
/// a.update_values(
///     |old, next| {
///         next.copy_from_slice(old);
///         next[1] = 5.0;
///         Ok::<_, SciOxideError>(())
///     },
///     &mut ws,
/// )?;
/// assert_eq!(a.values(), &[1.0, 5.0]);
/// assert_eq!(ws.check(&needed)?, Capacity::Sufficient);
/// // A new value left unassigned (NaN) is rejected, and the matrix is unchanged.
/// let unassigned = a.update_values(
///     |_, next| {
///         next[0] = 0.0;
///         Ok::<_, SciOxideError>(())
///     },
///     &mut ws,
/// );
/// assert_eq!(unassigned.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
/// assert_eq!(a.values(), &[1.0, 5.0]);
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseValueWorkspace<S: NumericScalar> {
    pub(crate) values: Vec<S>,
}
impl<S: NumericScalar> Default for SparseValueWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> SparseValueWorkspace<S> {
    /// Creates an empty workspace. Reserve it before updating a matrix.
    pub const fn new() -> Self {
        Self { values: Vec::new() }
    }
    /// Returns the bytes of reserved storage, including unused capacity.
    pub fn retained_bytes(&self) -> usize {
        self.values.capacity() * core::mem::size_of::<S>()
    }
    /// Reports whether the workspace can hold the requirement `r`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte size of `r` overflows `usize`.
    pub fn check(&self, r: &SparseValueRequirements) -> Result<Capacity, SciOxideError> {
        SparseValueRequirements::new::<S>(r.len)?;
        Ok(if self.values.capacity() >= r.len {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to hold the requirement `r`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte size of `r` overflows `usize`.
    /// - `AllocationFailure` if the storage cannot grow; the workspace keeps
    ///   its previous storage.
    pub fn try_reserve(&mut self, r: &SparseValueRequirements) -> Result<(), SciOxideError> {
        SparseValueRequirements::new::<S>(r.len)?;
        self.values
            .try_reserve_exact(r.len.saturating_sub(self.values.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("sparse_value_workspace"),
                    r.len,
                )
            })?;
        self.values.resize(r.len.max(self.values.len()), S::ZERO);
        Ok(())
    }
    pub(crate) fn update<R, E: From<SciOxideError>>(
        &mut self,
        accepted: &mut [S],
        fill: impl FnOnce(&[S], &mut [S]) -> Result<R, E>,
    ) -> Result<R, E> {
        crate::helpers::minimum(self.values.capacity(), accepted.len())?;
        self.values.resize(accepted.len(), S::ZERO);
        self.values
            .fill(S::from_real(<S::Real as scioxide_core::RealScalar>::NAN));
        let result = fill(accepted, &mut self.values)?;
        crate::helpers::finite_with_origin(&self.values, NonFiniteOrigin::Callback)?;
        // Equal lengths and `Copy` scalars: publication cannot fail or unwind,
        // and the workspace keeps its reserved capacity.
        accepted.copy_from_slice(&self.values);
        Ok(result)
    }
}

macro_rules! impl_value_updates {
    ($name:ident, $first:ident $(.$rest:ident)*) => {
        impl<S: scioxide_core::NumericScalar> $name<S> {
            /// Returns the workspace size that
            /// [`update_values`](Self::update_values) needs: one scalar per
            /// stored value.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if the byte size overflows `usize`.
            pub fn workspace_requirements(&self, (): ()) -> Result<crate::SparseValueRequirements, scioxide_core::SciOxideError> {
                crate::values::SparseValueRequirements::new::<S>(self.values().len())
            }
            /// Creates a workspace for [`update_values`](Self::update_values)
            /// on this matrix.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if the byte size overflows `usize`.
            /// - `AllocationFailure` if the storage cannot be allocated.
            pub fn workspace(&self, request: ()) -> Result<crate::SparseValueWorkspace<S>, scioxide_core::SciOxideError> {
                let mut workspace = crate::SparseValueWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(request)?)?;
                Ok(workspace)
            }
            /// Replaces all stored values with values computed by `fill`,
            /// keeping the sparsity pattern.
            ///
            /// `fill(old, new)` receives the current values and a slice of the
            /// same length, in storage order, set to NaN. It must write a finite
            /// value to every position of `new`. The new values are checked
            /// and then copied into the matrix, so on any error, or if `fill`
            /// panics, the matrix is unchanged. With a reserved workspace the
            /// call does not allocate.
            ///
            /// ```
            /// use scioxide_core::SciOxideError;
            /// use scioxide_sparse::CsrMatrix;
            ///
            /// let mut a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
            /// let mut ws = a.workspace(())?;
            /// a.update_values(
            ///     |old, next| {
            ///         next[0] = old[0] * 3.0;
            ///         Ok::<_, SciOxideError>(())
            ///     },
            ///     &mut ws,
            /// )?;
            /// assert_eq!(a.values(), &[6.0]);
            /// # Ok::<(), SciOxideError>(())
            /// ```
            ///
            /// # Errors
            ///
            /// - The error returned by `fill`, unchanged.
            /// - `DimensionMismatch` if `workspace` holds fewer scalars than
            ///   the matrix has stored values.
            /// - `NonFinite` (origin `Callback`) if a new value is NaN or
            ///   infinite, including a position that `fill` did not write.
            pub fn update_values<R, E: From<scioxide_core::SciOxideError>>(
                &mut self,
                fill: impl FnOnce(&[S], &mut [S]) -> Result<R, E>,
                workspace: &mut crate::SparseValueWorkspace<S>,
            ) -> Result<R, E> {
                workspace.update(&mut self.$first$(.$rest)*, fill)
            }
            /// Replaces all stored values with `values`, given in storage order.
            ///
            /// Unlike [`update_values`](Self::update_values), this needs no
            /// workspace. The values are checked before any is copied.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `values` does not have one entry per
            ///   stored value.
            /// - `NonFinite` if a value is NaN or infinite.
            pub fn copy_values_from(&mut self, values: &[S]) -> Result<(), scioxide_core::SciOxideError> {
                crate::helpers::exact(self.values().len(), values.len())?;
                crate::helpers::finite(values)?;
                self.$first$(.$rest)*.copy_from_slice(values);
                Ok(())
            }
            /// Replaces the stored value at `position`, an index into
            /// `values()`. The sparsity pattern does not change.
            ///
            /// # Errors
            ///
            /// - `OutOfBounds` if `position` is not less than the number of
            ///   stored values.
            /// - `NonFinite` if `value` is NaN or infinite.
            pub fn set_value(&mut self, position: usize, value: S) -> Result<(), scioxide_core::SciOxideError> {
                if position >= self.values().len() {
                    return Err(scioxide_core::SciOxideError::out_of_bounds(scioxide_core::quantity!("sparse_values"), position, self.values().len()));
                }
                crate::helpers::finite(&[value])?;
                self.$first$(.$rest)*[position] = value;
                Ok(())
            }
            /// Returns the bytes of the value storage, including unused
            /// capacity.
            pub fn retained_value_bytes(&self) -> usize {
                self.$first$(.$rest)*.capacity() * core::mem::size_of::<S>()
            }
        }
    };
}
pub(crate) use impl_value_updates;
