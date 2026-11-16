//! Mesh identity: mesh families and generation numbers.

use alloc::sync::Arc;
use core::fmt;

use scioxide_core::{InvalidInputReason, SciOxideError};

/// Identity of a mesh family within the running process.
///
/// A mesh family is a mesh and every revision derived from it. Two `MeshId`s
/// are equal exactly when one was cloned from the other; the comparison costs
/// O(1). Identities of independently constructed meshes are never equal while
/// both exist.
#[derive(Clone)]
#[must_use]
pub struct MeshId {
    anchor: Arc<()>,
}

impl MeshId {
    pub(crate) fn fresh() -> Self {
        Self {
            anchor: Arc::new(()),
        }
    }
}

impl PartialEq for MeshId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.anchor, &other.anchor)
    }
}

impl Eq for MeshId {}

impl fmt::Debug for MeshId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MeshId(..)")
    }
}

/// Revision number of a mesh within its family. Each revision that changes
/// geometry or topology increases it by one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct MeshGeneration(u64);

impl MeshGeneration {
    /// The generation of a newly constructed mesh, zero.
    pub const INITIAL: Self = Self(0);

    /// Returns the generation number.
    pub const fn get(self) -> u64 {
        self.0
    }

    // Generation
    pub(crate) fn next(self) -> Result<Self, SciOxideError> {
        self.0.checked_add(1).map(Self).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("mesh_generation"))
        })
    }

    #[cfg(test)]
    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// Identifies one revision of a mesh: its family and generation.
///
/// Fields, boundary sets and topology maps store the token of the mesh they
/// were built for. Comparing tokens costs O(1) and does not read coordinates
/// or connectivity.
///
/// ```
/// use scioxide_spatial::{AxisTopology, CoordinateAxis, Grid1d};
/// let grid = Grid1d::new(
///     [CoordinateAxis::uniform(0.0_f64, 1.0, 3)?],
///     [AxisTopology::Bounded],
/// )?;
/// let clone = grid.clone();
/// grid.mesh_token().ensure_compatible(clone.mesh_token())?;
/// let refined = grid.refined([2])?;
/// assert!(grid.mesh_token().same_family(refined.mesh_token()));
/// assert!(
///     grid.mesh_token()
///         .ensure_compatible(refined.mesh_token())
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct MeshToken {
    id: MeshId,
    generation: MeshGeneration,
    revision: Arc<()>,
}

// `generation` alone cannot tell sibling revisions apart. Two functional
// updates from the same parent both reach generation g + 1 but describe
// different meshes. The per-revision anchor makes them incompatible, while
// clones of one revision still compare equal in O(1).
impl PartialEq for MeshToken {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.generation == other.generation
            && Arc::ptr_eq(&self.revision, &other.revision)
    }
}
impl Eq for MeshToken {}

impl MeshToken {
    pub(crate) fn fresh() -> Self {
        Self {
            id: MeshId::fresh(),
            generation: MeshGeneration::INITIAL,
            revision: Arc::new(()),
        }
    }

    /// Returns the mesh family.
    pub const fn id(&self) -> &MeshId {
        &self.id
    }

    /// Returns the generation.
    pub const fn generation(&self) -> MeshGeneration {
        self.generation
    }

    /// Returns whether two tokens belong to the same mesh family.
    pub fn same_family(&self, other: &Self) -> bool {
        self.id == other.id
    }

    /// Returns whether two tokens identify the exact same revision.
    /// Sibling revisions are incompatible even when their generations match.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    /// Checks that two tokens identify the exact same revision.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the revisions differ, including sibling revisions.
    pub fn ensure_compatible(&self, other: &Self) -> Result<(), SciOxideError> {
        if self.is_compatible_with(other) {
            Ok(())
        } else {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("other"),
                InvalidInputReason::Inconsistent,
                "mesh revisions must match",
            ))
        }
    }

    // Checked successor revision.
    pub(crate) fn next_generation(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            id: self.id.clone(),
            generation: self.generation.next()?,
            revision: Arc::new(()),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_generation_for_test(&self, generation: MeshGeneration) -> Self {
        Self {
            id: self.id.clone(),
            generation,
            revision: Arc::clone(&self.revision),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cloned_tokens_keep_exact_identity() {
        let token = MeshToken::fresh();
        let cloned = token.clone();
        assert!(token.same_family(&cloned));
        assert!(token.is_compatible_with(&cloned));
    }

    #[test]
    fn independent_tokens_never_share_identity() {
        let left = MeshToken::fresh();
        let right = MeshToken::fresh();
        assert!(!left.same_family(&right));
        assert!(!left.is_compatible_with(&right));
    }

    #[test]
    fn advancing_preserves_family_and_changes_generation() {
        let token = MeshToken::fresh();
        let advanced = token.next_generation().unwrap();
        assert!(token.same_family(&advanced));
        assert!(!token.is_compatible_with(&advanced));
        assert_eq!(advanced.generation().get(), 1);
    }

    #[test]
    fn generation_overflow_is_typed() {
        let token = MeshToken::fresh().with_generation_for_test(MeshGeneration::from_raw(u64::MAX));
        assert!(
            matches!(token.next_generation(), Err(error) if match error.view() { scioxide_core::SciOxideErrorView::SizeOverflow {  context: _view_context } => _view_context == scioxide_core::quantity!("mesh_generation"), _ => false })
        );
    }
}
