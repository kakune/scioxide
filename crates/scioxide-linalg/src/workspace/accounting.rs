//! Byte accounting and reservation for the workspaces of factorizations,
//! operator actions and Krylov recycling.

use crate::{LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::{Capacity, NumericScalar, SciOxideError};

pub(crate) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize, SciOxideError> {
    values.into_iter().try_fold(0usize, |a, b| {
        scioxide_core::backend::checked_add(
            a,
            b,
            scioxide_core::quantity!("action_workspace_bytes"),
        )
    })
}
pub(crate) fn product(values: &[usize]) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_product(
        values,
        scioxide_core::quantity!("action_workspace_elements"),
    )
}
pub(crate) fn bytes<T>(count: usize) -> Result<usize, SciOxideError> {
    product(&[count, size_of::<T>()])
}

/// Heap capacity currently held, including nested owned buffers.
pub(crate) trait StorageBytes {
    fn storage_bytes(&self) -> Result<usize, SciOxideError>;
}
impl<T> StorageBytes for Vec<T> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        bytes::<T>(self.capacity())
    }
}
impl<T: StorageBytes> StorageBytes for Option<T> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        self.as_ref().map_or(Ok(0), StorageBytes::storage_bytes)
    }
}
impl<S: NumericScalar> StorageBytes for Matrix<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        bytes::<S>(self.capacity())
    }
}
impl<S: NumericScalar> StorageBytes for LinalgWorkspace<'_, S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        self.retained_bytes()
    }
}

/// Inspect actual buffers, accumulating only the storage that must grow.
#[derive(Default)]
pub(crate) struct StoragePlan {
    bytes: usize,
    shortfall: Option<(usize, usize)>,
}
impl StoragePlan {
    pub(crate) fn new(retained: usize) -> Self {
        Self {
            bytes: retained,
            shortfall: None,
        }
    }
    pub(crate) fn buffer<T>(
        &mut self,
        buffer: &Vec<T>,
        required: usize,
    ) -> Result<(), SciOxideError> {
        self.capacity::<T>(buffer.capacity(), required)
    }
    pub(crate) fn matrix<S: NumericScalar>(
        &mut self,
        matrix: &Matrix<S>,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        self.capacity::<S>(matrix.capacity(), product(&[rows, cols])?)
    }
    pub(crate) fn capacity<T>(
        &mut self,
        available: usize,
        required: usize,
    ) -> Result<(), SciOxideError> {
        if available < required {
            self.shortfall.get_or_insert((required, available));
            self.bytes = sum([self.bytes, bytes::<T>(required - available)?])?;
        }
        Ok(())
    }
    pub(crate) fn dense<S: NumericScalar>(
        &mut self,
        workspace: &LinalgWorkspace<'_, S>,
        requirements: &LinalgRequirements,
    ) -> Result<(), SciOxideError> {
        if self.shortfall.is_none() {
            self.shortfall = workspace.capacity_shortfall(requirements);
        }
        let current = workspace.storage_bytes()?;
        self.bytes = sum([
            self.bytes,
            workspace.reservation_bytes(requirements)? - current,
        ])?;
        Ok(())
    }
    pub(crate) fn check(&self, limit: Option<usize>) -> Result<Capacity, SciOxideError> {
        crate::workspace::admit_bytes(
            self.bytes,
            limit,
            scioxide_core::quantity!("action_workspace_bytes"),
        )?;
        Ok(if self.shortfall.is_none() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    pub(crate) fn require(&self, limit: Option<usize>) -> Result<(), SciOxideError> {
        crate::workspace::admit_bytes(
            self.bytes,
            limit,
            scioxide_core::quantity!("action_workspace_bytes"),
        )?;
        if let Some((required, available)) = self.shortfall {
            return Err(SciOxideError::dimension_mismatch([required], [available]));
        }
        Ok(())
    }
}

/// Grow a buffer to `len`, reporting allocation failure as an error.
pub(crate) fn resize<T: Clone>(
    buffer: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), SciOxideError> {
    scioxide_core::backend::try_reserve(
        buffer,
        len.saturating_sub(buffer.len()),
        scioxide_core::quantity!("action_workspace_elements"),
    )?;
    buffer.resize(len, value);
    Ok(())
}

/// The byte limit of a family's requirements.
pub(crate) trait RequirementBytes {
    fn byte_limit(&self) -> Option<usize>;
}

/// Execution options, nested dense requirements and checked total bytes shared
/// by every [`requirements!`] type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StorageRequirements<S: NumericScalar> {
    pub(crate) execution: LinalgExecutionOptions,
    pub(crate) dense: [LinalgRequirements; 2],
    pub(crate) complex: [LinalgRequirements; 2],
    pub(crate) bytes: usize,
    scalar: PhantomData<S>,
}
impl<S: NumericScalar> StorageRequirements<S> {
    /// Check the family's own bytes plus both same-scalar dense workspaces against the byte limit.
    pub(crate) fn new(
        execution: LinalgExecutionOptions,
        bytes: usize,
        dense: [LinalgRequirements; 2],
        complex: [LinalgRequirements; 2],
    ) -> Result<Self, SciOxideError> {
        let bytes = sum([
            bytes,
            dense[0].required_bytes::<S>()?,
            dense[1].required_bytes::<S>()?,
        ])?;
        crate::workspace::admit_bytes(
            bytes,
            execution.byte_limit(),
            scioxide_core::quantity!("action_workspace_bytes"),
        )?;
        Ok(Self {
            execution,
            dense,
            complex,
            bytes,
            scalar: PhantomData,
        })
    }
}

/// Define a public requirements type over a crate-private layout.
///
/// The body names `NumericScalar` unqualified, so invoking modules import it;
/// the public-surface inventory expands this macro and cannot bind `$crate`.
macro_rules! requirements {
    ($name:ident, $layout:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy)]
        pub struct $name<S: NumericScalar> {
            pub(crate) layout: $layout,
            pub(crate) storage: crate::workspace::accounting::StorageRequirements<S>,
        }
        impl<S: NumericScalar> $name<S> {
            pub(crate) fn new(
                layout: $layout,
                execution: crate::LinalgExecutionOptions,
                bytes: usize,
                dense: [crate::LinalgRequirements; 2],
                complex: [crate::LinalgRequirements; 2],
            ) -> core::result::Result<Self, scioxide_core::SciOxideError> {
                Ok(Self {
                    layout,
                    storage: crate::workspace::accounting::StorageRequirements::new(
                        execution, bytes, dense, complex,
                    )?,
                })
            }
            /// Returns the bytes a new workspace needs, including nested storage.
            pub const fn required_bytes(&self) -> usize {
                self.storage.bytes
            }
            /// Returns the byte limit on the workspace, if any.
            pub const fn byte_limit(&self) -> Option<usize> {
                self.storage.execution.byte_limit()
            }
        }
        impl<S: NumericScalar> crate::workspace::accounting::RequirementBytes for $name<S> {
            fn byte_limit(&self) -> Option<usize> {
                self.byte_limit()
            }
        }
    };
}
pub(crate) use requirements;

/// Family-specific storage that shares one check-and-reserve procedure.
pub(crate) trait PlannedStorage<S: NumericScalar>: StorageBytes {
    type Requirements: RequirementBytes;
    fn storage_plan(&self, requirements: &Self::Requirements)
    -> Result<StoragePlan, SciOxideError>;
    fn configure(&mut self, requirements: &Self::Requirements) -> Result<(), SciOxideError>;
    fn check(&self, requirements: &Self::Requirements) -> Result<Capacity, SciOxideError> {
        self.storage_plan(requirements)?
            .check(requirements.byte_limit())
    }
    fn reserve(&mut self, requirements: &Self::Requirements) -> Result<(), SciOxideError> {
        let _ = self.check(requirements)?;
        self.configure(requirements)?;
        crate::workspace::admit_bytes(
            self.storage_bytes()?,
            requirements.byte_limit(),
            scioxide_core::quantity!("action_workspace_bytes"),
        )
    }
    fn prepare(&mut self, requirements: &Self::Requirements) -> Result<(), SciOxideError> {
        self.storage_plan(requirements)?
            .require(requirements.byte_limit())?;
        self.configure(requirements)
    }
}
